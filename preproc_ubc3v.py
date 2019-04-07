import os
from PIL import Image
import h5py
import numpy as np

def load_itop_side(base_dir):
	f_train_x = h5py.File(os.path.join(base_dir,'ITOP_side_train_depth_map.h5'),'r')
	f_train_y = h5py.File(os.path.join(base_dir,'ITOP_side_train_labels.h5'),'r')
	f_test_x = h5py.File(os.path.join(base_dir,'ITOP_side_test_depth_map.h5'),'r')
	f_test_y = h5py.File(os.path.join(base_dir,'ITOP_side_test_labels.h5'),'r')
	
	valid_mask_train = f_train_y['is_valid'][()].astype(bool)
	valid_mask_test = f_test_y['is_valid'][()].astype(bool)
	
	train_x = f_train_x['data'][()][valid_mask_train]
	train_y = f_train_y['segmentation'][()][valid_mask_train]
	test_x = f_test_x['data'][()][valid_mask_test]
	test_y = f_test_y['segmentation'][()][valid_mask_test]
	
	print(train_x.shape,train_y.shape)
	print(test_x.shape,test_y.shape)
	return train_x, train_y, test_x, test_y
		
# train_x, train_y, test_x, test_y = load_itop_side("/home/jsmith/itop")


color_to_bp_map = np.array([
    [255, 106, 0],
    [255, 0, 0],
    [255, 178, 127],
    [255, 127, 127],
    [182, 255, 0],
    [218, 255, 127],
    [255, 216, 0],
    [255, 233, 127],
    [0, 148, 255],
    [72, 0, 255],
    [48, 48, 48],
    [76, 255, 0],
    [0, 255, 33],
    [0, 255, 255],
    [0, 255, 144],
    [178, 0, 255],
    [127, 116, 63],
    [127, 63, 63],
    [127, 201, 255],
    [127, 255, 255],
    [165, 255, 127],
    [127, 255, 197],
    [214, 127, 255],
    [161, 127, 255],
    [107, 63, 127],
    [63, 73, 127],
    [63, 127, 127],
    [109, 127, 63],
    [255, 127, 237],
    [127, 63, 118],
    [0, 74, 127],
    [255, 0, 110],
    [0, 127, 70],
    [127, 0, 0],
    [33, 0, 127],
    [127, 0, 55],
    [38, 127, 0],
    [127, 51, 0],
    [64, 64, 64],
    [73, 73, 73],
    [0, 0, 0],
    [191, 168, 247],
    [192, 192, 192],
    [127, 63, 63],
	[127, 116, 63],
], dtype=np.int32)



def load_ubc3v(base_dir):
    dir_path_template = os.path.join(
        base_dir, "{section}", "{subsection}", "images", "{xy}", "{camera}")
    full_path_template = os.path.join(
        dir_path_template, "mayaProject.{idx:06d}.png")

    def extract_section(section):
        print('working on {} data'.format(section))
        print(os.listdir(os.path.join(base_dir,section)))

        subsections = [int(direct) for direct in os.listdir(
            os.path.join(base_dir, section)) if os.path.isdir(os.path.join(base_dir,section,direct))]
        subsections.sort()
        x = []
        y = []
        for subsection in subsections:
            print('subsection: {}'.format(subsection))
            print('len(x): {}'.format(len(x)))
            print('len(y): {}'.format(len(y)))
            for xy in ['depthRender', 'groundtruth']:
                print(xy)
                for camera in ['Cam1', 'Cam2', 'Cam3']:
                    print(camera)
                    num_fps = len(os.listdir(dir_path_template.format(section=section, subsection=subsection, xy=xy, camera=camera)))
                    for i in range(num_fps):
                        fp = full_path_template.format(section=section,subsection=subsection,xy=xy,camera=camera,idx=i+1)
                        # print('Image Number: {}'.format(i))
                        img = np.asarray(Image.open(fp))
                        if xy == 'depthRender':
                            x.append(img[:,:,0].astype(np.float16)/255.0 * (8 - .5) + .5)
                        elif xy == 'groundtruth':
                            res = np.zeros((424,512), dtype=np.uint8)
                            alpha = img[:,:,3] == 255
                            red = img[:,:,0]
                            green = img[:,:,1]
                            blue = img[:,:,2]
                            for i in range(len(color_to_bp_map)):
                                mask = alpha & (np.abs(red - color_to_bp_map[i,0]) <= 3) & (np.abs(green - color_to_bp_map[i,1]) <= 3) & (np.abs(blue - color_to_bp_map[i,2]) <= 3)
                                res[mask] = i+1
                            del mask
                            del alpha
                            del red
                            del green
                            del blue
                            y.append(res)
                        del img

        return np.array(x), np.array(y)
    train_x, train_y = extract_section('train')
    test_x, test_y = extract_section('test')
    valid_x, valid_y = extract_section('valid')
    return train_x, train_y, test_x, test_y, valid_x, valid_y

#train_x,train_y,test_x,test_y,valid_x,valid_y = load_ubc3v('/home/jsmith/ubc3v/hard_pose')

#np.savez('ubc3v_data.npz',train_x=train_x,train_y=train_y,test_x=test_x,test_y=test_y,valid_x=valid_x,valid_y=valid_y)


def load_ubc3v_subsection(base_dir, section, subsection):
    color_error = 10
    subsection = str(subsection)
    dir_path_template = os.path.join(
        base_dir, "{section}", "{subsection}", "images", "{xy}", "{camera}")
    full_path_template = os.path.join(
        dir_path_template, "mayaProject.{idx:06d}.png")
    assert(os.path.isdir(os.path.join(base_dir, section, subsection)))
    x = []
    y = []
    for xy in ['depthRender', 'groundtruth']:
        print(xy)
        for camera in ['Cam1', 'Cam2', 'Cam3']:
            print(camera)
            num_fps = len(os.listdir(dir_path_template.format(section=section, subsection=subsection, xy=xy, camera=camera)))
            for i in range(num_fps):
                fp = full_path_template.format(section=section,subsection=subsection,xy=xy,camera=camera,idx=i+1)
                # print('Image Number: {}'.format(i))
                tmp = Image.open(fp)
                #print(tmp.mode)
                img = np.asarray(tmp)
                #print(img.shape)
                if xy == 'depthRender':
                    mask = img[:,:,0] != 0
                    scaledImg = img[:,:,0].astype(np.float16)/255.0 * (8 - .5) + .5
                    x.append(scaledImg*mask)
                elif xy == 'groundtruth':
                    res = np.zeros((424,512), dtype=np.uint8)
                    alpha = img[:,:,3] == 255
                    red = img[:,:,0].astype(np.int32)
                    green = img[:,:,1].astype(np.int32)
                    blue = img[:,:,2].astype(np.int32)
                    for i in range(len(color_to_bp_map)):
                        mask = alpha & (np.abs(red - color_to_bp_map[i,0]) <= color_error) & (np.abs(green - color_to_bp_map[i,1]) <= color_error) & (np.abs(blue - color_to_bp_map[i,2]) <= color_error)
                        res[mask] = i+1
                    y.append(res)
    return np.array(x), np.array(y)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", default="train")
    parser.add_argument("-ss", type=int)
    parser.add_argument("-p", default="easy")
    args = parser.parse_args()
    print(args.p)
    print(args.s)
    print(args.ss)
    tx, ty = load_ubc3v_subsection('/home/jsmith/ubc3v/{}_pose'.format(args.p),args.s,args.ss)
    np.savez('{}_{}_{:04d}.npz'.format(args.p,args.s,args.ss), x=tx, y=ty)




    
