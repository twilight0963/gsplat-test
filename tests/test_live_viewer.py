import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from src.live_viewer import PreviewPublisher, FilePreview
from src import engine


class LiveViewerTests(unittest.TestCase):
    def test_publisher_finishes_without_closing_viewer_and_next_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader=FilePreview(tmp)
            publisher=PreviewPublisher(tmp)
            publisher.publish({'means':torch.ones(2,3)}, 1, 2)
            data,status,error=reader.read()
            self.assertTrue(data.pop('_new_session'))
            torch.testing.assert_close(data['means'],torch.ones(2,3))
            self.assertIn('50.0%',status)
            publisher.publish({'means':torch.full((2,3),2.)},2,2)
            publisher.finish();publisher.close()
            data,status,error=reader.read()
            self.assertFalse(data.pop('_new_session'))
            self.assertIn('model saved',status)
            torch.testing.assert_close(data['means'],torch.full((2,3),2.))
            self.assertIsNone(reader.read()[0])
            self.assertFalse(reader.closed)
            another=PreviewPublisher(tmp)
            self.assertIsNone(reader.read()[0])
            another.publish({'means':torch.zeros(3,3)},0,3)
            data,_,_=reader.read()
            self.assertTrue(data['_new_session'])
            self.assertEqual(data['means'].shape,(3,3))

    def test_failed_training_leaves_viewer_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader=FilePreview(tmp);publisher=PreviewPublisher(tmp)
            publisher.publish({'means':torch.ones(1,3)},0,1)
            publisher.finish(RuntimeError('test failure'))
            _,status,error=reader.read()
            self.assertIn('test failure',status)
            self.assertIsNone(error)

    def test_engine_returns_after_export(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(engine,'extract_frames'), patch.object(engine,'run_colmap'), \
             patch.object(engine,'load_reconstruction',return_value=({},[],32,32)), \
             patch.object(engine,'PreviewPublisher') as publisher, \
             patch.object(engine,'ensure_viewer') as ensure, \
             patch.object(engine,'train_splats',return_value={}), \
             patch.object(engine,'export_gltf') as export:
            output=Path(tmp)
            result=engine.build_model(Path('input.mp4'),output)
            self.assertEqual(result,output/'model.glb')
            export.assert_called_once_with({},result)
            ensure.assert_called_once()
            publisher.return_value.finish.assert_called_once_with()
            publisher.return_value.close.assert_called_once()
