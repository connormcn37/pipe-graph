# pipe-graph
node graph editor for data (video) pipelines

## Concept:

a program for video data pipeline processing. The base class which most objects share inheritance is named `Entity`, which at minimum has: 
  - a string `label` such that no two entities are allowed to have the same label, thus it can be used as unique id 
  - `inputs`: a list of input labels referencing other entity instances 
  - `connect()`: tells the entity to (re)initialize from input labels 
  - `disconnect()`: release pointers, stop 

A `Stage` is an `Entity` with extra properties: 

  - `parameters`: a dictionary of key:value pairs used by that stage to tune whatever transformation is applied to the input data. 

`Stage` also has functions:
  - `get_last_frame()`: returns most recent output frame created by this stage. Used for "pull" based data flow 
  - `push_frame()`: used for "push" based data flow 

There are several different classes representing different kinds of stages such as: 
  - `CropStage` (crop input frame), 
  - `CastStage` (change data type between float,uint8), 
  - `SplitStage` (given an input frame of shape (width, height, k) create k outputs (width,height) )
  - `MergeStage` (given k inputs of shape (width,height) create 1 output of shape (width,height,k) 

A `Pipeline` is an `Entity` that can be initialized by giving a list of stage types with labels and parameters to initialize them and connect them together. It additionally has the functions: 

  - `start()`: orchestrates the stages data processing sequentially 
  - `stop()`: stop processing.

```
