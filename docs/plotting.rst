========
Plotting
========

.. plot::
   :context: close-figs
   :include-source:

    import numpy as np
    import tensorflow as tf

    import matplotlib.pyplot as plt
    import seaborn as sns

    import os.path
    import glob

    from functools import partial
    from observations.util import maybe_download_and_extract

    # Constants declarations
    observed_height, observed_width, observed_channels = observed_shape = [128, 128, 4]

    img_grid_rows = 8
    img_grid_cols = 5

    buffer_size = 5000
    batch_size = img_grid_rows * img_grid_cols

    # Function declarations
    def fetch_pokemon_data(path='../datasets'):

        url = "https://github.com/PokeAPI/sprites/archive/master.zip"
        path = os.path.expanduser(path)

        if not os.path.exists(os.path.join(path, 'sprites')):

            maybe_download_and_extract(path, url, save_file_name='sprites.zip')

        return glob.glob(os.path.join(path, 'sprites/sprites/pokemon/*.png'))

    def normalize_image(image, shape):

        image_height, image_width, image_channels = shape

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=image_height,
                                                       target_width=image_width)

        return image

    def load_image(filename, shape):

        image_height, image_width, image_channels = shape

        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=image_channels)

        return image

    def dataset_from_filenames(filenames, target_shape, path='../datasets'):

        path = os.path.expanduser(path)

        dataset = tf.data.Dataset.from_tensor_slices(filenames) \
                                 .map(partial(load_image, shape=target_shape)) \
                                 .map(partial(normalize_image, shape=target_shape))

        return dataset

    def plot_image_grid(ax, images, shape, n=20, m=None):
        """
        Plot the first `n * m` vectors in the array as 
        a `n`by`m` grid of `img_rows`by`img_cols` images.
        """
        if m is None:
            m = n

        cmap = 'gray' if shape[-1] == 1 else None

        grid = images[:n*m].reshape(n, m, *shape).squeeze()

        return ax.imshow(np.vstack(np.dstack(grid)), cmap=cmap)

    train_dataset = dataset_from_filenames(fetch_pokemon_data('../datasets'),
                                           target_shape=observed_shape) \
        .shuffle(buffer_size=buffer_size) \
        .batch(batch_size, drop_remainder=True)

    train_iterator = train_dataset.make_one_shot_iterator()
    x_data = train_iterator.get_next()

    fig, ax = plt.subplots(figsize=golden_size(20))

    with tf.Session() as sess:
        plot_image_grid(ax, sess.run(x_data), observed_shape, 
                        n=img_grid_rows, m=img_grid_cols)

    plt.show()