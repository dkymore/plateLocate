import pl
import cv2
import os

work_dir = input("work_dir:")
if not work_dir: work_dir = r"E:\Projects\22.9.27_ML\Imgs\car"
output_dir = input("output_dir:")
if not output_dir:output_dir = r"E:\Projects\22.9.27_ML\Imgs\output"
for root,dirs,files in os.walk(work_dir):
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.join(root,file)
            print(f"detecting {img_path}")
            plate_res = file.rstrip(".jpg")
            rl = pl.workflow(img_path)
            if len(rl)==1:
                cv2.imencode(f'{plate_res}.jpg',rl[0])[1].tofile(os.path.join(output_dir,plate_res+'.jpg'))
                #cv2.imwrite(os.path.join(output_dir,plate_res+'.jpg'),rl[0])
            