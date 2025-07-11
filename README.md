Source code for "Practical Deep Learning: A Python-Based Introduction"
----------------------------------------------------------------------
(first edition)

### Code for the second edition is [here](https://github.com/rkneusel9/PracticalDeepLearning2E)

You'll find the source code included or referenced in the book in this
archive.  The code is organized by chapter.  If the chapter is not listed,
there was no code to go with it.

All the code is Python 3.X and requires the libraries installed in Chapter 1
of the book.

Please send questions, comments, or bugs to:

    rkneuselbooks@gmail.com

Updates:

    page 84: the URL for the Iris dataset has changed:
                https://archive.ics.uci.edu/dataset/53/iris

    page 86: the URL for the Breast Cancer dataset has changed:
                https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original

    TensorFlow issues addressable with:
                pip3 install tensorflow-cpu  (TF 2.8)
                then update repo code to move 'keras' imports to 'tensorflow.keras'

    Moving from Adadelta to Adam:
                Adadelta appears to be broken in newer versions of TensorFlow.  Therefore, if you
                are getting poor performance, I suggest moving all models to Adam as the optimizer.
                Simply replace "Adadelta" with "Adam" and you should be good to go.

    The file tutorial.pdf is a beginner's guide to NumPy, SciPy, Matplotlib, and Pillow.

