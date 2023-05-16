import numpy as np
import matplotlib.pyplot as plt

def plot_ct(image, index):
    """
    image: np.array of iamge
    index: list index have mask value
    """
    num_fig = int(np.ceil(np.sqrt(len(index))))

    fig, axs = plt.subplots(num_fig, num_fig, figsize=(20,14))

    slide = 0
    for x in range(num_fig):
        for y in range(num_fig):
            try:
                axs[x, y].imshow(image[index[slide]])
                axs[x, y].set_title(index[slide])
                slide += 1
            except:
                pass
            
    plt.show()