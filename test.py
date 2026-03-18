# from pathlib import Path

# import cv2
# import numpy as np

# pt = [[93, 65], [234, 65], [45, 165], [278, 165]]
# desire_pos = [[[50, 50], [100, 50], [50, 100], [100, 100]], 
#               [[100, 100], [150, 100], [100, 150], [150, 150]],
#               [[200, 200], [300, 200], [200, 300], [300, 300]]]

# def main() -> None:
#   base_dir = Path(__file__).resolve().parent
#   image_path = base_dir / "img.png"

#   image = cv2.imread(str(image_path))
#   if image is None:
#     raise FileNotFoundError(f"Could not read image: {image_path}")

#   cv2.imshow("Ori", image)

#   for i in range(len(desire_pos)):
#     src_pts = np.array(pt, dtype=np.float32)
#     dst_pts = np.array(desire_pos[i], dtype=np.float32)
#     homography = cv2.getPerspectiveTransform(src_pts, dst_pts)

#     warped = cv2.warpPerspective(image, homography, (500, 500))
#     cv2.imshow(f"Warped{i}", warped)
    
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()

# if __name__ == "__main__":
#   main()

from pathlib import Path

import cv2
import numpy as np

pt = [[93, 65], [234, 65], [45, 165], [278, 165]]
desire_pos = [[[150, 180], [250, 180], [150, 280], [250, 280]]]
roi = [[[175, 205], [225, 205], [175, 255], [225, 255]]] #
final_res = (390, 300)

def main() -> None:
  base_dir = Path(__file__).resolve().parent
  image_path = base_dir / "img.png"

  image = cv2.imread(str(image_path))
  if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

  cv2.imshow("Ori", image)

  for i in range(len(desire_pos)):
    src_pts = np.array(pt, dtype=np.float32)
    dst_pts = np.array(desire_pos[i], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(image, homography, final_res)
    cv2.putText(warped, f"{final_res[0]}x{final_res[1]}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if i < len(roi):
      r = roi[i]
      tl = (r[0][0], r[0][1])
      br = (r[3][0], r[3][1])
      cv2.rectangle(warped, tl, br, (0, 255, 0), 2)
      w = br[0] - tl[0]
      h = br[1] - tl[1]
      cv2.putText(warped, f"{w}x{h}", (tl[0], tl[1] - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow(f"Warped{i}", warped)
    
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()