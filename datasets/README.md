
For a few datasets that detectron2 natively supports,
the datasets are assumed to exist in a directory called
"datasets/", under the directory where you launch the program.
They need to have the following directory structure:

## Expected dataset structure for cityscapes:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note:
labelTrainIds.png are created by `cityscapesscripts/preparation/createTrainIdLabelImgs.py`.
They are not needed for instance segmentation.

## Expected dataset structure for Pascal VOC:
```
VOC20{07,12}/
  Annotations/
  ImageSets/
  JPEGImages/
```
