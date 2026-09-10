import datetime
from time import sleep

import cv2


def startup_cam():
    capture = cv2.VideoCapture(0)
    _ = capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    _ = capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _ = capture.set(cv2.CAP_PROP_FPS, 15)
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("Requested 15, got:", fps)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    out = cv2.VideoWriter(
        f"Recorded_vid_{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5.5))).strftime('%d_%m_%y_%H_%M')}_{fps}.mp4",
        fourcc,  # pyright: ignore[reportUnknownArgumentType]
        fps,
        (frame_width, frame_height),
        True,
    )
    print("Starting camera...")
    return capture, out


capture = None
while True:
    capture, out = startup_cam()
    if not capture:
        print("Waiting for camera connection...")
        capture, out = startup_cam()
        sleep(5)
        continue
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        cv2.imshow("Camera", frame)
        out.write(frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()
    break
