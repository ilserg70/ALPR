# -*- coding: utf-8 -*-
from pathlib import Path
import math

import cv2
import numpy as np
import imutils

import utils

# This file contains a set of image transformations
# based on: https://docs.opencv.org/master/

IMG_HEIGHT =  64
IMG_WIDTH  = 128

def do_random_transformations(img_path, weight, out_dir, min_score=0.2, min_br=60):
    """ Execute random transformations.
        @param: img_path  - Path to image file
        @param: weight    - How many images is need to generate
        @param: out_dir   - Output directory to save generated images
        @param: min_score - Min value of score threshold for generated images
        @param: min_br    - Min value of brightness for generated images
        @return: Count of generated images
    """
    helper = {'count': 0, 'required': weight, 'attempts': 0, 'max_attempts': 10 * weight, 'msg': '', 'files': []}

    info = utils.split_name(img_path)

    if info['score'] < min_score:
        helper['msg'] = f"score: {info['score']} < min_score: {min_score}"
        return helper

    img = imread(img_path)
    img = resize(img, IMG_WIDTH, IMG_HEIGHT)
    
    blur_score = calc_blurring_score(img)
    if blur_score == 0:
        helper['msg'] = f"blur_score: {blur_score}"
        return helper

    area_score = info['score']/blur_score
    img_ = resize(img, w=2*IMG_WIDTH, h=2*IMG_HEIGHT)

    # List of image transformations with names
    trans = [
        ('mot', 2, random_add_motion_blur),
        ('rot', 3, random_rotate),
        ('r3d', 1, random_rotate_3D_and_warp),
        ('blr', 1, random_add_gaussian_blur),
        ('shd', 1, random_adding_shade),
        ('noi', 1, random_add_salt_and_pepper_noises),
        ('cut', 1, random_cutting),
        ('pad', 1, random_padding),
        ('tbc', 1, random_top_bottom_cutting),
        ('brg', 1, random_brightness_adjustment),
        ('rsz', 1, random_downsize),
        #-----------------------------
        # ('dlt', 1, random_add_dilation),
        # ('ers', 1, random_add_erosion),
        # ('opn', 1, random_add_opening),
        # ('cls', 1, random_add_closing),
        # ('grd', 1, random_add_gradient),
        # ('eql', 1, equalize_hist)
    ]
    trans_weights = np.array([t[1] for t in trans])
    trans_weights = trans_weights / trans_weights.sum()

    min_choice, max_choice = 2, 5
    if len(trans)<5:
        min_choice, max_choice = 1, len(trans)

    img_hash = set()
    indices = list(range(len(trans)))
    while helper['count'] < helper['required'] and helper['attempts'] < helper['max_attempts']:
        out = img_
        for i in np.random.choice(indices, np.random.randint(min_choice, max_choice), p=trans_weights, replace=False):
            _, _, func = trans[i]
            out = func(out)
        out = resize(out, w=IMG_WIDTH, h=IMG_HEIGHT)
        score_ = area_score * calc_blurring_score(out)
        br = brightness(out)
        h = calc_img_hash(out)
        if score_ >= min_score and br>=min_br and h not in img_hash:
            helper['count'] += 1 # 1-based numeration
            info_ = info.copy()
            info_['score'] = score_
            info_['idx'] = helper['count']
            out_path = out_dir.joinpath(utils.mk_file_name(info_))
            imwrite(out, out_path)
            img_hash.add(h)
            helper['files'].append(str(out_path))
        helper['attempts'] += 1
    return helper

def do_random_transformations2(img_path, weight, out_dir, min_score=0.2, min_br=60):
    """ Execute random transformations.
        @param: img_path  - Path to image file
        @param: weight    - How many images is need to generate
        @param: out_dir   - Output directory to save generated images
        @param: min_score - Min value of score threshold for generated images
        @param: min_br    - Min value of brightness for generated images
        @param: selected_trans - List of image transform. func. codes: 'mot,rot,r3d,blr,shd,noi,cut,pad,tbc,brg,rsz'
        @return: Count of generated images
    """
    helper = {'count': 0, 'required': weight, 'attempts': 0, 'max_attempts': 10 * weight, 'msg': '', 'files': []}

    info = utils.split_name(img_path)

    if info['score'] < min_score:
        helper['msg'] = f"score: {info['score']} < min_score: {min_score}"
        return helper

    img = imread(img_path)
    img = resize(img, IMG_WIDTH, IMG_HEIGHT)
    
    blur_score = calc_blurring_score(img)
    if blur_score == 0:
        helper['msg'] = f"blur_score: {blur_score}"
        return helper

    area_score = info['score']/blur_score
    img_ = resize(img, w=2*IMG_WIDTH, h=2*IMG_HEIGHT)

    # List of image transformations with names
    trans = [
            ('rotate', random_rotate, 0.45),
            ('brightness', random_brightness_adjustment, 0.45),
            ('cutting_padding', random_cutting_or_padding, 0.2),
            ('cutting', random_top_bottom_cutting, 0.2),
            ('noises', random_add_salt_and_pepper_noises, 0.2),
            ('blur', random_add_gaussian_blur, 0.15),
            ('shade', random_adding_shade, 0.2),
            ('motion', random_add_motion_blur, 0.1),
            ('downsize', random_downsize, 0.1),
            # ('rotate_3d', random_rotate_3D_and_warp, 0.05),
            #----------------------------------------------------
            # ('dilation', random_add_dilation),
            # ('erosion', random_add_erosion),
            # ('opening', random_add_opening),
            # ('closing', random_add_closing),
            # ('gradient', random_add_gradient),
            # ('equalize', equalize_hist)
        ]

    img_hash = set()
    while helper['count'] < helper['required'] and helper['attempts'] < helper['max_attempts']:
        out = img_
        for i in range(len(trans)):
            _, func, prob = trans[i]
            if np.random.rand() < prob:
                out = func(out)
        
        out = resize(out, w=IMG_WIDTH, h=IMG_HEIGHT)
        score_ = area_score * calc_blurring_score(out)
        br = brightness(out)
        h = calc_img_hash(out)
        if score_ >= min_score and br>=min_br and h not in img_hash:
            helper['count'] += 1 # 1-based numeration
            info_ = info.copy()
            info_['score'] = score_
            info_['idx'] = helper['count']
            out_path = out_dir.joinpath(utils.mk_file_name(info_))
            imwrite(out, out_path)
            img_hash.add(h)
            helper['files'].append(str(out_path))
        helper['attempts'] += 1
    return helper

def gen_image(img_path, out_dir, params, count_map):
    """ Generate augmented image. (Hang's function with no changes.)
        New image will be saved if new score not below threshold
        @param: img_path - Path to input image
        @param: out_dir - Path to output augmented images
        @param: params - {
            'text': str, 'state': str, 'score': float, 'frame_id': str, 
            'stacked': ('[' in text), 
            'all_stacked': (text[0] == '[' or text[-1] == ']'), 
            'no_warn': alphabet.no_warn(text), 
            'hist_equalize': False
        }
        @param: count_map - Count of generated images for each input file
        @return: None
    """
    out = imread(img_path)

    blur_score = calc_blurring_score(out)
    area_score = params['score']/blur_score

    cut_factor = 0.025 if params['all_stacked'] else 0.035
    if np.random.rand() < 0.7:
        out = random_cutting_or_padding(out, factor=cut_factor)

    # Brightness adjustment
    if np.random.rand() > 0.25:
        out = random_brightness_adjustment(out)

    out = out/ 255.0
    out = resize(out, w=IMG_WIDTH, h=IMG_HEIGHT)

    # Rotate
    if np.random.rand() > 0.1:
        max_rotate = 3.5 if params['all_stacked'] else 4.5
        out = random_rotate(out, max_angle=max_rotate)

    out = resize(out, 2*IMG_WIDTH, 2*IMG_HEIGHT)

    #  Add Gaussian blur
    rad = 3 if params['no_warn'] and not params['stacked'] else 2 if params['no_warn'] or not params['stacked'] else 1
    if np.random.rand() > 0.5:
        out = random_add_gaussian_blur(out, rad=rad)

    if  params['score'] > 0.5:
        #     Add salt and pepper noises
        if np.random.rand() > 0.35:
            sp_scale = 0.075 if params['stacked'] else 0.2
            out = random_add_salt_and_pepper_noises(out, scale=sp_scale)

    #     Add Gaussian blur 2nd
        if np.random.rand() > 0.5:
            out = random_add_gaussian_blur(out, rad=rad)

    # #     Add dilation
    #     if np.random.rand() > 0.9 and params['no_warn']:
    #         out = random_add_dilation(out)
    #
    # #     Add Erosion
    #     if np.random.rand() > 0.9 and params['no_warn']:
    #         out = random_add_erosion(out)

    #     Add motion blur
        if np.random.rand() > 0.5 and params['no_warn']:
            out = random_add_motion_blur(out)

    # Random resize to increase blur
    if np.random.rand() > 0.5 and params['no_warn'] and not params['stacked']:
        out = random_resize(out)

    # More Random top/bottom boundary cut
    out = random_top_bottom_cutting(out)

    # Adding shade
    if np.random.rand() > 0.75:
        out = random_adding_shade(out)

    out = np.round(out * 255.).astype(np.uint8)

    if params['hist_equalize']:
        print('warning: using equalizeHist')
        out = equalize_hist(out)

    score_ = area_score * calc_blurring_score(out)
    thres = 0.2 if params['stacked'] else 0.05

    if score_ >= thres:
        if str(img_path) not in count_map:
            count_map[str(img_path)] = 0
        count_map[str(img_path)] += 1 # One based numeration of augmented images
        idx = count_map[str(img_path)]
        out_file = f"{params['text']:s}_{str(params['state']):s}_{score_:.3f}_{params['frame_id']:s}_{idx:03d}.jpg"
        out_path = out_dir.joinpath(out_file)
        imwrite(out, out_path)

def imread(img_path):
    try:
        return cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite(img, img_path):
    try:
        cv2.imwrite(str(img_path), img)
        return Path(img_path).exists()
    except Exception:
        return False

def read_image(image_path):
    try:
        image_bin = open(str(image_path), 'rb') .read()
        return image_bin if check_image_is_valid(image_bin) else None
    except Exception:
        return None

def resize(img, w, h):
    """ Image resizing
        @param: w - Image width
        @param: h - Image height
        @return: image
    """
    h_, w_ = img.shape[0], img.shape[1]
    if h_ == h and w_ == w:
        return img
    # interpolation - cv2.INTER_LINEAR (default) cv2.INTER_AREA cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

def resize_image(img, factor):
    if factor and factor != 1:
        h, w, _ = img.shape
        h_new, w_new = int(h * factor), int(w * factor)
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
    return img

def concat_images(files, m, out_path=None):
    """ Concatenate images to square m x m images. If len(files) < m*m, random select the rest from files.
    @param: files - Input image files
    @param: m - Numb. of images by vert. and horiz.
    @param: out_path - Output image path
    @return: image - Result image
    """
    ff = files + np.random.choice(files, m*m-len(files))
    out = cv2.vconcat([cv2.hconcat([resize(imread(f), IMG_WIDTH, IMG_HEIGHT) for f in p]) for p in list(utils.chunks(ff, m))[:m]])
    if out_path:
        imwrite(out, out_path)
    return out

def check_image_is_valid(image_bin):
    img_h, img_w = 0, 0
    try:
        if image_bin is not None:
            # Note: np.fromstring(...) will make a copy of the string in memory, while: np.frombuffer(...) 
            # will use the memory buffer of the string directly and won't use any additional memory.
            # image_buf = np.fromstring(image_bin, dtype=np.uint8)
            image_buf = np.frombuffer(image_bin, dtype=np.uint8)
            img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
            img_h, img_w = img.shape[0], img.shape[1]
    except Exception:
        return False
    return img_h * img_w != 0

def calc_img_hash(img, hash_size=8):
    # resize the input image, adding a single column (width) so we can compute the horizontal gradient
    img_ = cv2.resize(img, (hash_size + 1, hash_size))
    # compute the (relative) horizontal gradient between adjacent column pixels
    diff = img_[:, 1:] > img_[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def brightness(img):
    """ Good brightness >= 30
    """
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(np.linalg.norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)

def warp(img, points):
    """ Get four corners of the billboard.
        @param: img - Frame image
        @param: points - 4 points of licence plate
        @return: plate image resized
    """
    try:
        pts_src = np.array(points, np.float32)
        pts_dst = np.array([(0, 0), (IMG_WIDTH, 0), (IMG_WIDTH, IMG_HEIGHT), (0, IMG_HEIGHT)], np.float32)
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        return cv2.warpPerspective(img.copy(), M, (IMG_WIDTH, IMG_HEIGHT))
    except Exception:
        return None

def calc_pixels_score(img):
    """ Calculate ration of non zero pixels.
    @param: img - Image
    @return: score
    """
    h, w = img.shape[0], img.shape[1]
    if h==0 or w==0:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.count_nonzero(gray)/h/w

def calc_blurring_score(img, min_blur=0.0, max_blur=8.0):
    """ Calculate Laplacian blurring score.
        Examples: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
            @return blurring_score in range [0.0; 1.0]
    """
    if img.mean()==0.0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var() ** 0.25
    if blur <= min_blur:
        return 0.0
    elif blur >= max_blur:
        return 1.0
    return (blur - min_blur) / (max_blur - min_blur)

def equalize_hist(img):
    """ Histogram equalization to improve the contrast of the image.
        https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
    """
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # Applying equalize Hist operation on Y channel.
    img_y_cr_cb[:,:,0] = cv2.equalizeHist(img_y_cr_cb[:,:,0])
    # convert back to RGB format
    return cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCR_CB2BGR)

def equalize_hist_2(img):
    """ Histogram equalization to improve the contrast of the image.
        https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def random_resize(img):
    size_ = int(round(np.random.rand()*32 + 32))
    return resize(img, 2*size_, size_)

def random_add_motion_blur(img):
    """ Random add motion blur effect and image rotation.
        https://subscription.packtpub.com/book/application_development/9781785283932/2/ch02lvl1sec21/motion-blur
        https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    """
    size = int(round(4 + np.random.rand() * 30))
    angle = np.random.rand() * 180
    h, w = img.shape[0], img.shape[1]    
    out = resize(img, w*2, h*2)
    out = imutils.rotate_bound(out, angle)
    # Generation the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    # Applying the kernel to the input image
    out = cv2.filter2D(out, -1, kernel_motion_blur)
    out = imutils.rotate_bound(out, -angle)
    s = list((np.round((np.array(out.shape) / 2))).astype(int))
    a, b = s[0], s[1]
    out = out[a-h+2 : a+h-2, b-w+2 : b+w-2]
    return out

def _random_adjust_gamma(img, inv_flag=False):
    """
        https://docs.opencv.org/master/d2/de8/group__core__array.html#gab55b8d062b7f5587720ede032d34156f
    """
    gamma = np.random.rand()*0.65 + 0.35
    if inv_flag:
        gamma = 1./gamma
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in np.arange(0, 256)
    ]).astype(np.uint8)
    return cv2.LUT(img, table)

def random_brightness_adjustment(img):
    """
        https://docs.opencv.org/master/d2/de8/group__core__array.html#gab55b8d062b7f5587720ede032d34156f
    """
    inv_flag=np.random.rand()>0.5
    return _random_adjust_gamma(img, inv_flag)

def random_rotate(img, max_angle=None):
    """ Image rotation at random angle.
    """

    def _rotate_image(img, angle):
        """
        Rotates an OpenCV 2 / np image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """
        h, w = img.shape[0], img.shape[1]
        h2, w2 = h*0.5, w*0.5
        x0, y0 = int(w2), int(h2) # image center
        
        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack([cv2.getRotationMatrix2D((x0, y0), angle, 1.0), [0, 0, 1]])
        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])
        
        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-w2,  h2]) * rot_mat_notranslate).A[0],
            (np.array([ w2,  h2]) * rot_mat_notranslate).A[0],
            (np.array([-w2, -h2]) * rot_mat_notranslate).A[0],
            (np.array([ w2, -h2]) * rot_mat_notranslate).A[0]
        ]
        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        y_coords = [pt[1] for pt in rotated_coords]
        new_w = int(abs(max(x_coords) - min(x_coords)))
        new_h = int(abs(max(y_coords) - min(y_coords)))
        
        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - w2)],
            [0, 1, int(new_h * 0.5 - h2)],
            [0, 0, 1]
        ])
        
        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
        
        # Apply the transform
        return cv2.warpAffine(img, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

    def _largest_rotated_rect(w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.
        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
        Converted to Python by Aaron Snoswell
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    def _crop_around_center(img, width, height):
        """
        Given a np / OpenCV 2 image, crops it to the given width and height,
        around it's centre point
        """
        h, w = img.shape[0], img.shape[1]
        if width<=w and height<=h:
            return img
        height, width = min(height, h), min(width, w)
        x0, y0 = int(w * 0.5), int(h * 0.5) # image center
        x1 = int(max([x0 - width  * 0.52, 0]))
        x2 = int(min([x0 + width  * 0.52, w]))
        y1 = int(max([y0 - height * 0.52, 0]))
        y2 = int(min([y0 + height * 0.52, h]))
        return img[y1:y2, x1:x2]

    if not max_angle:
        max_angle = 3.5 if np.random.rand()>0.5 else 4.5
    angle = (np.random.rand() * 2 - 1) * max_angle
    h, w = img.shape[0], img.shape[1]
    out = _rotate_image(img, angle)
    out = _crop_around_center(out, *_largest_rotated_rect(w, h, math.radians(angle)))
    return resize(out, IMG_WIDTH, IMG_HEIGHT)

def random_cutting_or_padding(img, factor=None):
    if np.random.rand() > 0.5:
        return random_cutting(img, factor)
    return random_padding(img, factor)

def random_cutting(img, factor=None):
    h, w = img.shape[0], img.shape[1]
    if not factor:
        factor = 0.025 if np.random.rand()>0.5 else 0.035
    left   = int(round(w * np.random.rand() * factor))
    right  = int(round(w * (1- np.random.rand() * factor)))
    top    = int(round(h * np.random.rand() * factor))
    bottom = int(round(h * (1- np.random.rand() * factor)))
    return img[top:bottom, left:right]

def random_padding(img, factor=None):
    h, w = img.shape[0], img.shape[1]
    if not factor:
        factor = 0.025 if np.random.rand()>0.5 else 0.035
    left   = int(round(w * np.random.rand() * 2 * factor))
    right  = int(round(w * np.random.rand() * 2 * factor))
    top    = int(round(h * np.random.rand() * 2 * factor))
    bottom = int(round(h * np.random.rand() * 2 * factor))
    borderType = cv2.BORDER_CONSTANT
    rng = range(5,175)
    padding_color = tuple(np.random.choice(rng).item() for _ in range(3))
    return cv2.copyMakeBorder(img, top, bottom, left, right, borderType, value=padding_color)

def random_add_gaussian_blur(img, rad=None):
    if not rad:
        x = np.random.rand()
        rad = 1 if x<0.33 else 2 if x<0.66 else 3
    dist = 2 * int(round(rad + np.random.rand() * rad)) + 1
    return cv2.GaussianBlur(img, (dist, dist), int(round(2 * np.random.rand())))

def random_add_salt_and_pepper_noises(img, scale=None):
    if not scale:
        scale = np.random.rand()/14
    max_val = 255 if img.max() > 1 else 1.
    out = img.copy() + max_val * np.random.normal(scale=scale, size=img.shape)
    return np.clip(out, 0, max_val).astype(np.uint8)

def random_downsize(img):
    """ Random downsize to increase blur.
    """
    size = int(round(np.random.rand()*32 + 32))
    return resize(img, 2*size, size)

def random_top_bottom_cutting(img):
    h = img.shape[0]
    h1, h2 = 0, h
    tmp = np.random.rand()
    if tmp < 0.25:
        h1 = int(h * 0.2 * np.random.rand())
    elif tmp < 0.6:
        h2 = int(h * (1 - 0.075 * np.random.rand()))
    out = img[h1:h2, :]
    return resize(out, IMG_WIDTH, IMG_HEIGHT)

def random_adding_shade(img):
    """ Adding shade randomly.
    """
    h, w = img.shape[0], img.shape[1]
    h1, w1, h2, w2 = 0, 0, h, w
    h_, w_ = int(h*np.random.rand()), int(w*np.random.rand())
    tmp = np.random.rand()
    if tmp < 0.25:
        h2, w2 = h_, w_
    elif tmp < 0.5:
        h2, w1 = h_, w_
    elif tmp < 0.75:
        h1, w1 = h_, w_
    else:
        h1, w2 = h_, w_
    try:
        shaded = img[h1:h2, w1:w2].copy()
        if img.max() > 1:
            shaded = random_brightness_adjustment(shaded)
        else:
            shaded = np.round(shaded * 255.).astype(np.uint8)
            shaded = random_brightness_adjustment(shaded)
            shaded = shaded / 255.0
        out = img.copy()
        out[h1:h2, w1:w2] = shaded
        return out
    except:
        return img

# Morphological Transformations.
# -------------------------------------------------
def _random_morphological_transformations(img, mode='dilation'):
    """ Morphological Transformations.
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
            @param: img  - Input image
            @param: mode - Type of transformation: 'dilation', 'erosion', 'opening', 'closing', 'gradient', 'tophat', 'blackhat'
            @return transformed image
    """
    dist = 2 * int(round(np.random.rand() * 1)) + 1
    kernel = np.ones((dist, dist), np.uint8)
    if mode=='dilation':
        return cv2.dilate(img, kernel, iterations=1)
    if mode=='erosion':
        return cv2.erode(img, kernel, iterations=1)
    if mode=='opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if mode=='closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if mode=='gradient':
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    if mode=='tophat':
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    if mode=='blackhat':
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

def random_add_dilation(img):
    """ Dilation transformation on binary image"""
    return _random_morphological_transformations(img, mode='dilation')

def random_add_erosion(img):
    """ Erosion transformation on binary image"""
    return _random_morphological_transformations(img, mode='erosion')

def random_add_opening(img):
    """ Opening transformation on binary image"""
    return _random_morphological_transformations(img, mode='opening')

def random_add_closing(img):
    """ Closing transformation on binary image"""
    return _random_morphological_transformations(img, mode='closing')

def random_add_gradient(img):
    """ Gradient transformation on binary image"""
    return _random_morphological_transformations(img, mode='gradient')

def random_add_tophat(img):
    """ Top Hat transformation on binary image"""
    return _random_morphological_transformations(img, mode='tophat')

def random_add_blackhat(img):
    """ Black Hat transformation on binary image"""
    return _random_morphological_transformations(img, mode='blackhat')

# Image Gradient Transformations.
# -------------------------------------------------
def _image_gradient(img, mode='laplacian'):
    """ Image Gradient Transformations.
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
            @param: mode - Type of transformation: 'laplacian', 'sobel_x', 'sobel_y'
    """
    if mode=='laplacian':
        return cv2.Laplacian(img, cv2.CV_64F).astype(np.uint8)
    if mode=='sobel_x':
        return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5).astype(np.uint8)
    if mode=='sobel_y':
        return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5).astype(np.uint8)

def gradient_laplacian(img):
    """Laplacian gradient transformations"""
    return _image_gradient(img, mode='laplacian')

def gradient_sobel_x(img):
    """Sobel horizontal gradient transformations (along X-axis)"""
    return _image_gradient(img, mode='sobel_x')

def gradient_sobel_y(img):
    """Sobel vertical gradient transformations (along Y-axis)"""
    return _image_gradient(img, mode='sobel_y')

# Image rotation around X or Y
# -------------------------------------------------

def matrix_projection_2D_to_3D(h, w):
    return np.array([
        [1, 0, -w/2],
        [0, 1, -h/2],
        [0, 0,    0],
        [0, 0,    1]  
    ], dtype=np.double)

def matrix_rotation_X(alpha_rad):
    c, s = np.cos(alpha_rad), np.sin(alpha_rad)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1]  
    ], dtype=np.double)

def matrix_rotation_Y(alpha_rad):
    c, s = np.cos(alpha_rad), np.sin(alpha_rad)
    return np.array([
        [c, 0, -s, 0],
        [0, 1,  0, 0],
        [s, 0,  c, 0],
        [0, 0,  0, 1]  
    ], dtype=np.double)

def matrix_translation_Z(dist):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dist],
        [0, 0, 0, 1]  
    ], dtype=np.double)

def matrix_projection_3D_to_2D(h, w, f):
    return np.array([
        [f, 0, w/2, 0],
        [0, f, h/2, 0],
        [0, 0,   1, 0] 
    ], dtype=np.double)

def mk_transfo(A2, T, R, A1):
    """ transfo = A2 * (T * (R * A1))
    """
    return np.tensordot(A2, np.tensordot(T, np.tensordot(R, A1, axes=1), axes=1), axes=1)

def get_dest_points(w, h, M):
    pts_src = np.array([[0, 0],[w, 0],[w, h],[0, h]], np.double)
    pts_src = np.array([pts_src])
    pts_dst = cv2.perspectiveTransform(pts_src, M).astype(np.int32)
    return pts_dst

def _draw_polygon(img, pts):
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], True, (0,255,255))

def rotation_3D(img, alpha_rad, flag_x, flag_warp=False, flag_show=False):
    focus=25
    dist=30
    h, w = img.shape[0], img.shape[1]
    A1 = matrix_projection_2D_to_3D(h,w)
    R  = matrix_rotation_X(alpha_rad) if flag_x else matrix_rotation_Y(alpha_rad)
    T  = matrix_translation_Z(dist)
    A2 = matrix_projection_3D_to_2D(h, w, focus)
    M = mk_transfo(A2, T, R, A1)
    out = cv2.warpPerspective(img.copy(), M, (w, h))
    if flag_warp or flag_show:
        pts_dst = get_dest_points(w, h, M)
        if flag_show:
            print(pts_dst)
            _draw_polygon(out, pts_dst)
        if flag_warp:
            out = warp(out, pts_dst)
    return out

def random_rotate_3D_and_warp(img):
    """ Random rotation around X or Y axes
    """
    alpha_rad = np.pi/(20 + 40 * np.random.rand())
    if np.random.rand()<0.5:
        alpha_rad = -alpha_rad
    flag_x = np.random.rand()<0.5    
    return rotation_3D(img, alpha_rad, flag_x, flag_warp=True, flag_show=False)
