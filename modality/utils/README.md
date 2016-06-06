# general library utilities

* Classes:
  * [DataManager](#DataManager): Abstract class object that reads data from file
  * [MnistManager](#MnistManager): Data manager object specifically for Mnist data
  * [Logger](#Logger): Creates a logger object that saves training loss information to file
* Simple utils:
  * [grid](#grid): A collection of functions for image manipulation

<a name="DataManager"></a>
## DataManager ##
```lua
data = DataManager(batchSize)
```
An abstract class that supports the following functionalities:
* `DataManager`
  * `inEpoch()`: checks whether we have completed an epoch of training
  * `shuffle(batchSize)`: shuffles the training set according to specificed batchSize
  * `next()`: gets the next minibatch
  * `cuda()`: send the training data to GPU

<a name="MnistManager"></a>
## MnistManager ##
```lua
data = MnistManager(batchSize)
```
A subclass of `DataManager` that deals specifically with the MNIST dataset.


<a name="Logger"></a>
## Logger ##
```lua
logger = Logger(name, iter)
```
A class that performs saving of training loss information to file and outputing the a running average of the training loss to stdout. It supports the following functionalities:
* `Logger`
  * `receiveRecord(comm, learningRate)`. Receives information about learning rate and comm. Comm is a table with two tables:
    * `comm.record`: A table containing a list of relevant training loss information from each iteration. i.e.:

      ```lua
      comm.record = {kldErr=10, predErr=20}
      ```
    * `comm.recordName`: The corresponding name of each record.

      ```lua
      comm.recordName = {'predErr', 'kldErr'}
      ```
      The order of the names in `recordName` specifics the order in which the records are printed to file/stdout.
  * `log`: when called, it prints the iteration number, the learning rate, and a running average of all the training loss information.

<a name="grid"></a>
## grid ##
* `grid` is a small library that provides the following functions:
  * `grid.stack(images nRow, nCol)`: Takes a batch of images (i.e. `nBatch x H x W` or `nBatch x C x H x W`) and converts it into a grid of size `H*nRow x W*nCol` or `C x H*nRow x W*nCol` (i.e. places the images side-by-side in a grid). The images are placed into the grid in order from left to right, starting at the top row.
  * `grid.t(image)`: Transpose the iamge along the last two dimensions
  * `grid.split(tensor, dim)`: Convenience function that splits and squeezes a tensor along a particular dimension.

    


