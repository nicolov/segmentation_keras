import numpy as np

pascal_nclasses = 21
pascal_palette = np.array([(0, 0, 0)
    , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
    , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
    , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
    , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)], dtype=np.uint8)


# 0=background
# 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
# 6=bus, 7=car, 8=cat, 9=chair, 10=cow
# 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


def mask_to_label(mask_rgb):
    """From color-coded RGB mask to classes [0-21]"""
    mask_labels = np.zeros(mask_rgb.shape[:2])

    for i in range(mask_rgb.shape[0]):
        for j in range(mask_rgb.shape[1]):
            mask_labels[i, j] = pascal_palette.index(tuple(mask_rgb[i, j, :].astype(np.uint8)))

    return mask_labels


def interp_map(prob, zoom, width, height):
    zoom_prob = np.zeros((height, width, prob.shape[2]), dtype=np.float32)
    for c in range(prob.shape[2]):
        for h in range(height):
            for w in range(width):
                r0 = h // zoom
                r1 = r0 + 1
                c0 = w // zoom
                c1 = c0 + 1
                rt = float(h) / zoom - r0
                ct = float(w) / zoom - c0
                v0 = rt * prob[r1, c0, c] + (1 - rt) * prob[r0, c0, c]
                v1 = rt * prob[r1, c1, c] + (1 - rt) * prob[r0, c1, c]
                zoom_prob[h, w, c] = (1 - ct) * v0 + ct * v1
    return zoom_prob
