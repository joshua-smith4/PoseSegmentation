import os
from PIL import Image
import h5py
import numpy as np

# color to body part mapping
# used to change rgb images to body part labels
# index of array corresponds to body part index
# 45 different colors for 45 different body parts
# rgba(0,0,0,0) is reserved for background
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


# loads and converts entire ubc3v dataset
# requires a lot of memory, close to 200 GB

# base_dir is directory where ubc3v is downloaded
def load_ubc3v(base_dir):
	# path template used to target specific sections, subsections, train/test, and camera
    dir_path_template = os.path.join(
        base_dir, "{section}", "{subsection}", "images", "{xy}", "{camera}")
	# full path to individual images
    full_path_template = os.path.join(
        dir_path_template, "mayaProject.{idx:06d}.png")

	# function used to extract a specific section ('train', 'test', or 'valid')
    def extract_section(section):
        print('working on {} data'.format(section))
        print(os.listdir(os.path.join(base_dir,section)))
		# each section is subdivided into subsections
		# this gets all subsections in section directory
        subsections = [int(direct) for direct in os.listdir(
            os.path.join(base_dir, section)) if os.path.isdir(os.path.join(base_dir,section,direct))]
        subsections.sort()
        x = []
        y = []
		# loop through all subsections
        for subsection in subsections:
            print('subsection: {}'.format(subsection))
            print('len(x): {}'.format(len(x)))
            print('len(y): {}'.format(len(y)))
			# loop through image and label data
            for xy in ['depthRender', 'groundtruth']:
                print(xy)
				# loop through each camera (same pose from 3 different angles)
                for camera in ['Cam1', 'Cam2', 'Cam3']:
                    print(camera)
                    num_fps = len(os.listdir(dir_path_template.format(section=section, subsection=subsection, xy=xy, camera=camera)))
					# get number of images in current subsection and camera
                    for i in range(num_fps):
						# construct file path to specific image data
                        fp = full_path_template.format(section=section,subsection=subsection,xy=xy,camera=camera,idx=i+1)
                        # open image and convert to numpy array
                        img = np.asarray(Image.open(fp))
                        if xy == 'depthRender':
							# case where image is not label data
							# use function provided in Matlab API for ubc3v here:
							# https://github.com/ashafaei/ubc3v
							# to convert from gray scale image to depth data
                            x.append(img[:,:,0].astype(np.float16)/255.0 * (8 - .5) + .5)
                        elif xy == 'groundtruth':
							# prepare label result array
                            res = np.zeros((424,512), dtype=np.uint8)
							# extract channels from input
							# and create alpha mask
							# anywhere where alpha is equal to zero is background
                            alpha = img[:,:,3] == 255
                            red = img[:,:,0]
                            green = img[:,:,1]
                            blue = img[:,:,2]
							# go through every color in map and label it accordingly in res
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
	# get each section and return values
    train_x, train_y = extract_section('train')
    test_x, test_y = extract_section('test')
    valid_x, valid_y = extract_section('valid')
    return train_x, train_y, test_x, test_y, valid_x, valid_y

# because previous function require so much memory
# this function was made to load one subsection at a time
def load_ubc3v_subsection(base_dir, section, subsection):
	# permitted error in each channel for matching color in label
	# images to body part color in color_to_bp_map
    color_error = 10
	# get subsection string, can be passed as an int
    subsection = str(subsection)
	# path template used to target specific sections, subsections, train/test, and camera
	dir_path_template = os.path.join(
		base_dir, "{section}", "{subsection}", "images", "{xy}", "{camera}")
	# full path to individual images
	full_path_template = os.path.join(
		dir_path_template, "mayaProject.{idx:06d}.png")
    assert(os.path.isdir(os.path.join(base_dir, section, subsection)))
    x = []
    y = []
	# loop image and label data
    for xy in ['depthRender', 'groundtruth']:
		# loop through each camera (same pose from 3 different angles)
        for camera in ['Cam1', 'Cam2', 'Cam3']:
            print(camera)
            num_fps = len(os.listdir(dir_path_template.format(section=section, subsection=subsection, xy=xy, camera=camera)))
            for i in range(num_fps):
				# get individual file paths
                fp = full_path_template.format(section=section,subsection=subsection,xy=xy,camera=camera,idx=i+1)
                tmp = Image.open(fp)
                img = np.asarray(tmp)
                if xy == 'depthRender':
					# get background and put in mask
                    mask = img[:,:,0] != 0
					# convert from grayscale to depth measure
                    scaledImg = img[:,:,0].astype(np.float16)/255.0 * (8 - .5) + .5
					# zero out background
                    x.append(scaledImg*mask)
                elif xy == 'groundtruth':
					# prepare result label array
                    res = np.zeros((424,512), dtype=np.uint8)
					# extract channels and create alpha mask
					# anywhere alpha is 0 is background
                    alpha = img[:,:,3] == 255
                    red = img[:,:,0].astype(np.int32)
                    green = img[:,:,1].astype(np.int32)
                    blue = img[:,:,2].astype(np.int32)
					# loop through color_to_bp_map and label pixels that match
					# as body parts (0 background, 1-45 body parts)
                    for i in range(len(color_to_bp_map)):
                        mask = alpha & (np.abs(red - color_to_bp_map[i,0]) <= color_error) & (np.abs(green - color_to_bp_map[i,1]) <= color_error) & (np.abs(blue - color_to_bp_map[i,2]) <= color_error)
                        res[mask] = i+1
                    y.append(res)
    return np.array(x), np.array(y)

# load and preprocess data when called as main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
	# section 'train', 'test', or 'valid'
    parser.add_argument("-s", default="train")
	# subsection to be processed
    parser.add_argument("-ss", type=int)
	# 'hard', 'inter', or 'easy' used to drill into the
	# three partitions in the dataset
    parser.add_argument("-p", default="easy")
	# base dir of ubc3v dataset
	parser.add_argument('-b')
    args = parser.parse_args()
    print(args.p)
    print(args.s)
    print(args.ss)
    tx, ty = load_ubc3v_subsection(os.path.join(args.b, '{}_pose').format(args.p),args.s,args.ss)
	# save extracted data in numpy array format
    np.savez('{}_{}_{:04d}.npz'.format(args.p,args.s,args.ss), x=tx, y=ty)
