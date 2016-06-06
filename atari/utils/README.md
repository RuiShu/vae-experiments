# general library utilities

* Classes:
  * [DataLoader](#MnistManager): Data manager object specifically for reading the atari data
  * [DataManager](#DataManager): A high-level data management object that manages a threadpool of DataLoaders
  * [Logger](#Logger): Creates a logger object that saves training loss information to file
  * [Saver](#Saver): A saver object that provides functionalities for reading and saving trained networks

<a name="DataManager"></a>
## DataManager ##
```lua
data = DataManager(nThread, batchSize, segLength, preprocess)
```
This class creates a threadpool of DataLoaders that continually read in a batch of image sequences (and preprocesses them).

<a name="DataLoader"></a>
## DataLoader ##
```lua
data = DataLoader(batchSize, segLength, preprocess)
```
A class that reads in a batch of image sequences (and preprocesses them) at each iteration. Simply call `data:next()` to read the next minibatch of data.

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

<a name="Saver"></a>
## Saver ##
```lua
saver = Saver(cmd, infoList)
```
A convenience object that can either:
* Take in commandline arguments (which specifies the model architecture and other hyperparameters) and constructs the appropriate network. Since the commandline arguments specifies additional information irrelavent to the actual network, `infoList` is an array containing the set of relevant keys that we also pass in.
* Read in an existing trained network to continue training

To save the network, simply called
```lua
saver:save(model, config, state, iter)
```
The `state` and `config` variables are parameters for the `optim` class and is necessary if you wish to resume training for a saved network. To counter data corruption issues if the program is interrupted during saving, a local backup is always saved subsequently.