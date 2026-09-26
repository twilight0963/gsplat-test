import base64
import io
import json
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from threading import Thread
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src import upload_server
from src.upload_server import Jobs, PHOTO_BATCH, make_server, receive_photos


def batch(*files):
    body = b''
    for name, data in files:
        encoded = name.encode()
        body += struct.pack('>I', len(encoded)) + encoded + struct.pack('>Q', len(data)) + data
    return body


class ReceivePhotosTests(unittest.TestCase):
    def receive(self, body, length=None):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        target = tmp / 'photos'
        receive_photos(io.BytesIO(body), len(body) if length is None else length, target)
        return target

    def test_stores_flat_photos_byte_for_byte(self):
        target = self.receive(batch(('DJI_0001.JPG', b'a' * 5), ('b ü.tiff', b'bb')))
        self.assertEqual(sorted(p.name for p in target.iterdir()), ['DJI_0001.JPG', 'b ü.tiff'])
        self.assertEqual((target / 'DJI_0001.JPG').read_bytes(), b'a' * 5)

    def test_rejects_unsafe_unsupported_and_duplicate_names(self):
        for name in ('../x.jpg', 'a/x.jpg', 'a\\x.jpg', '.x.jpg', 'x.txt', 'Thumbs.db', 'x\0.jpg'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.receive(batch((name, b'1'), ('ok.jpg', b'2')))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.receive(batch(('a.jpg', b'1'), ('A.JPG', b'2')))

    def test_rejects_malformed_and_short_bodies(self):
        good = batch(('a.jpg', b'12'), ('b.jpg', b'34'))
        cases = {'truncated': (good[:-1], len(good)), 'extra bytes': (good + b'x', len(good) + 1),
                 'size past end': (good[:-2], len(good) - 2), 'one photo': (batch(('a.jpg', b'1')), None),
                 'empty file': (batch(('a.jpg', b''), ('b.jpg', b'1')), None)}
        for label, (body, length) in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                self.receive(body, length)


class HttpPhotoUploadTests(unittest.TestCase):
    def setUp(self):
        self.jobs = Jobs()
        self.jobs.launch = Mock()
        self.server = make_server('127.0.0.1', 0, self.jobs)
        self.thread = Thread(target=self.server.serve_forever)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.uploads = upload_server.ROOT / 'runs' / '.uploads'
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.before = set(self.uploads.iterdir())

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, body, kind, query=''):
        request = Request(f'{self.base}/upload?output=runs/test-{uuid.uuid4().hex}{query}', data=body,
                          headers={'Content-Type': kind})
        with urlopen(request, timeout=5) as response:
            return response.status

    def test_photo_batch_launches_engine_on_directory(self):
        self.assertEqual(self.post(batch(('a.jpg', b'1'), ('b.jpg', b'22')), PHOTO_BATCH, '&photo-every=3&every=9'), 202)
        path, params = self.jobs.launch.call_args.args
        self.addCleanup(shutil.rmtree, path)
        self.assertTrue(path.is_dir())
        self.assertEqual(path.parent, self.uploads)
        self.assertEqual((path / 'b.jpg').read_bytes(), b'22')
        self.assertEqual(params['photo-every'], '3')
        self.assertNotIn('every', params)

    def test_video_upload_does_not_pass_photo_every(self):
        self.assertEqual(self.post(b'video', 'application/octet-stream'), 202)
        path, params = self.jobs.launch.call_args.args
        self.addCleanup(path.unlink)
        self.assertEqual(params['every'], '5')
        self.assertNotIn('photo-every', params)

    def test_bad_batch_is_reported_and_removed(self):
        with self.assertRaises(HTTPError) as caught:
            self.post(batch(('a.jpg', b'1'), ('notes.txt', b'2')), PHOTO_BATCH)
        self.assertEqual(caught.exception.code, 400)
        self.assertIn('notes.txt', json.load(caught.exception)['error'])
        self.assertEqual(set(self.uploads.iterdir()), self.before)
        self.jobs.launch.assert_not_called()
        self.assertFalse(self.jobs.snapshot()['busy'])

    def test_browser_disconnect_is_reported_quietly_and_cleaned_up(self):
        for kind in (PHOTO_BATCH, 'application/octet-stream'):
            with self.subTest(kind=kind):
                output = io.StringIO()
                with redirect_stdout(output), socket.create_connection(('127.0.0.1', self.server.server_port)) as sock:
                    # Announce 1 MB, send a fragment, then close as a reloaded page would.
                    sock.sendall((f'POST /upload?output=runs/test-{uuid.uuid4().hex} HTTP/1.1\r\n'
                                  f'Host: x\r\nContent-Type: {kind}\r\nContent-Length: 1000000\r\n\r\n').encode()
                                 + batch(('a.jpg', b'x' * 100))[:50])
                    sock.shutdown(socket.SHUT_WR)
                    sock.recv(4096)
                self.assertIn('Upload interrupted: the browser closed the connection', output.getvalue())
                self.assertNotIn('Traceback', output.getvalue())
                self.assertIn('Upload interrupted', self.jobs.snapshot()['status'])
                self.assertFalse(self.jobs.snapshot()['busy'])
                self.assertEqual(set(self.uploads.iterdir()), self.before)
        self.jobs.launch.assert_not_called()

    def test_reply_to_a_closed_connection_does_not_raise(self):
        handler = object.__new__(self.server.RequestHandlerClass)
        handler.wfile = Mock(write=Mock(side_effect=BrokenPipeError))
        handler.send_response = handler.send_header = handler.end_headers = Mock()
        handler.reply(400, b'{}')


@unittest.skipUnless(shutil.which('node'), 'node is required to run the upload page script')
class PageScriptTests(unittest.TestCase):
    """Runs the page's own photo-selection and batch code, decoding the result in Python."""

    def run_page(self, mode, entries):
        html = (upload_server.ROOT / 'src' / 'upload.html').read_text()
        script = re.search(r'// photo-upload-start\n(.*)// photo-upload-end', html, re.S).group(1)
        program = script + r"""
const [mode, entries] = [process.argv[1], JSON.parse(process.argv[2])];
const makeFile = (path, data, relative) => {
  const file = new File([data], path.split('/').pop(), {type: path.endsWith('.mp4') ? 'video/mp4' : ''});
  Object.defineProperty(file, 'webkitRelativePath', {value: relative ? path : ''});
  return file;
};
// A FileSystemDirectoryEntry for a dropped folder; readEntries() returns batches of 100 like Chromium.
function dirEntry(name, children) {
  return {isDirectory: true, isFile: false, name, createReader() {
    let at = 0;
    return {readEntries(ok) { setTimeout(() => { const out = children.slice(at, at + 100); at += out.length; ok(out); }); }};
  }};
}
const fileEntry = (name, data) => ({isFile: true, isDirectory: false, name, file(ok) { ok(makeFile(name, data)); }});
function dropped() {  // entries like ["Images/", [["a.jpg", "1"], ["sub/", []]]]
  return entries.map(([name, value]) => name.endsWith('/')
    ? dirEntry(name.slice(0, -1), value.map(([n, v]) => n.endsWith('/') ? dirEntry(n.slice(0, -1), []) : fileEntry(n, v)))
    : fileEntry(name, value));
}
async function select() {
  if (mode === 'input') return {files: directPhotos(entries.map(([p, d]) => makeFile(p, d, true)))};
  if (mode === 'handle') {
    const children = entries.map(([p, d]) => p.includes('/') ? {kind: 'directory', name: p.split('/')[0]}
                                                              : {kind: 'file', name: p, getFile: async () => makeFile(p, d)});
    return {files: await handlePhotos({values: async function* () { yield* children; }})};
  }
  if (mode === 'drop') {
    const items = dropped();
    const files = items.filter(i => i.isFile).map(i => makeFile(i.name, ''));
    return readDrop(items, files);
  }
  return {files: entries.map(([p, d]) => makeFile(p, d))};
}
select().then(async result => {
  let picked;
  try {
    if (result.mode === 'video') { console.log(JSON.stringify({mode: 'video', names: result.files.map(f => f.name)})); return; }
    picked = pickPhotos(result.files);
  } catch (error) { console.log(JSON.stringify({error: error.message})); return; }
  const buffer = await photoBatch(picked).arrayBuffer();
  console.log(JSON.stringify({mode: result.mode, folder: result.name, names: picked.map(f => f.name),
                              body: Buffer.from(buffer).toString('base64')}));
}, error => console.log(JSON.stringify({error: error.message})));
"""
        out = subprocess.run(['node', '-e', program, mode, json.dumps(entries)],
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    def decode(self, result):
        body = base64.b64decode(result['body'])
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        receive_photos(io.BytesIO(body), len(body), tmp / 'p')
        return {p.name: p.read_text() for p in (tmp / 'p').iterdir()}

    def test_dropped_folder_reads_every_batch_and_only_direct_photos(self):
        children = [[f'DJI_{i:04d}.JPG', str(i)] for i in range(250)] + [['Thumbs.db', 'x'], ['.hidden.jpg', 'x'], ['sub/', []]]
        result = self.run_page('drop', [['Images/', children]])
        self.assertEqual(result['mode'], 'photos')
        self.assertEqual(result['folder'], 'Images')
        self.assertEqual(len(result['names']), 250)  # three readEntries() batches
        decoded = self.decode(result)
        self.assertEqual(decoded['DJI_0000.JPG'], '0')
        self.assertEqual(decoded['DJI_0249.JPG'], '249')

    def test_dropped_dataset_root_is_refused_with_guidance(self):
        # Dataset/ATR: a preview image and notes, with the photos one level down in Images/.
        result = self.run_page('drop', [['ATR/', [['ATR.png', 'p'], ['Metadata.txt', 'm'], ['Images/', []]]]])
        self.assertIn('Found 1 JPEG, PNG or TIFF photos', result['error'])
        self.assertIn('Dataset/ATR/Images', result['error'])

    def test_dropped_video_or_photos_and_invalid_drops(self):
        self.assertEqual(self.run_page('drop', [['flight.mp4', 'v']]), {'mode': 'video', 'names': ['flight.mp4']})
        self.assertEqual(self.run_page('drop', [['b.png', '1'], ['a.jpg', '2']])['names'], ['a.jpg', 'b.png'])
        self.assertIn('one folder at a time', self.run_page('drop', [['A/', []], ['B/', []]])['error'])
        self.assertIn('one folder at a time', self.run_page('drop', [['A/', []], ['a.jpg', '1']])['error'])
        self.assertIn('one video at a time', self.run_page('drop', [['a.mp4', 'v'], ['b.mp4', 'v']])['error'])

    def test_folder_pickers_use_direct_photos_only(self):
        result = self.run_page('handle', [['DJI_2.jpg', 'second'], ['DJI_1.JPG', 'first ü'], ['Thumbs.db', 'x'], ['sub/x.jpg', 'n']])
        self.assertEqual(self.decode(result), {'DJI_1.JPG': 'first ü', 'DJI_2.jpg': 'second'})
        result = self.run_page('input', [['Images/b.png', '1'], ['Images/a.tif', '2'], ['Images/sub/c.jpg', '3']])
        self.assertEqual(result['names'], ['a.tif', 'b.png'])

    def test_selected_files_and_errors(self):
        self.assertEqual(self.run_page('files', [['a.jpeg', '1'], ['b.jpg', '2'], ['c.txt', '3']])['names'], ['a.jpeg', 'b.jpg'])
        self.assertIn('Found 0', self.run_page('handle', [])['error'])
        self.assertIn('share the name', self.run_page('files', [['a.jpg', '1'], ['A.JPG', '2']])['error'])
