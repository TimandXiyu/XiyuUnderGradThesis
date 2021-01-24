from PIL import Image
import os
import tqdm

if __name__ == "__main__":
    base = r'C:\Users\Tim Wang\Desktop\gitclone\XiyuUnderGradThesis'
    dirlist = os.listdir(os.path.join(base, r'data\fake_data_6.0\fake_mask'))
    for name in tqdm.tqdm(dirlist):
        img_path = os.path.join(r'C:\Users\Tim Wang\Desktop\gitclone\XiyuUnderGradThesis\data\fake_data_6.0\fake_mask', name)
        img = Image.open(img_path).convert('L')
        img = img.resize((1024, 1024))
        img.save(img_path)
