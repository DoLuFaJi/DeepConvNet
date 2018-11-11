import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 36, 36

def show_dataset(dataset, n=6):
    show_images = 10
    for i in range(show_images):
        print(dataset[i]['is_face'])
    img = np.vstack((np.hstack((np.asarray(dataset[i]['image']) for _ in range(n)))
                   for i in range(show_images)))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
