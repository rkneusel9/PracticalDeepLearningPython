Source code for the fully convolutional experiments in Chapter 13
-----------------------------------------------------------------

The first set of experiments use the standard MNIST data.  Run
the code in this order:

1. mnist_cnn.py (to create 'mnist_cnn_base_model.h5')
2. mnist_cnn_fcn.py (to create 'mnist_cnn_fcn_model.h5')
3. make_large_mnist_test_images.py (to create larger multidigit images)
4. mnist_cnn_fcn_single_digits.py (to verify the model works with single digits)
5. mnist_cnn_fcn_test_large.py (to classify the large images)
6. mnist_cnn_fcn_graymaps.py (to make heat maps)


The second set of experiments uses an augmented and shifted MNIST dataset.
Run the code in this order:

1. mnist_cnn_aug.py (makes 'mnist_cnn_base_aug_model.h5')
2. mnist_cnn_aug_fcn.py (makes 'mnist_cnn_aug_fcn_model.h5')
3. mnist_cnn_aug_fcn_test.py (tests single images & large images, 'results_aug')
4. mnist_cnn_aug_fcn_heatmaps.py (creates 'heatmaps_aug')

