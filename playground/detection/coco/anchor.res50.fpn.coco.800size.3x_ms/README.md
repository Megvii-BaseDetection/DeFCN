# anchor.res50.fpn.coco.800size.3x_ms  

seed: 10266195

## Evaluation results for bbox:  

```  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.371
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.538
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.406
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.235
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.468
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.550
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.605
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.407
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.642
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.752
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 37.121 | 53.845 | 40.606 | 23.457 | 40.621 | 46.818 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 52.613 | bicycle      | 27.819 | car            | 41.466 |  
| motorcycle    | 38.400 | airplane     | 60.967 | bus            | 62.903 |  
| train         | 58.376 | truck        | 30.933 | boat           | 22.104 |  
| traffic light | 25.824 | fire hydrant | 64.486 | stop sign      | 62.992 |  
| parking meter | 36.029 | bench        | 19.416 | bird           | 32.151 |  
| cat           | 61.740 | dog          | 53.929 | horse          | 53.916 |  
| sheep         | 47.808 | cow          | 53.603 | elephant       | 58.405 |  
| bear          | 64.446 | zebra        | 67.133 | giraffe        | 64.894 |  
| backpack      | 13.931 | umbrella     | 35.645 | handbag        | 13.484 |  
| tie           | 29.557 | suitcase     | 34.717 | frisbee        | 60.803 |  
| skis          | 18.115 | snowboard    | 26.962 | sports ball    | 47.191 |  
| kite          | 41.637 | baseball bat | 21.392 | baseball glove | 32.027 |  
| skateboard    | 45.930 | surfboard    | 28.968 | tennis racket  | 42.757 |  
| bottle        | 34.439 | wine glass   | 33.286 | cup            | 38.131 |  
| fork          | 26.456 | knife        | 14.392 | spoon          | 12.133 |  
| bowl          | 36.752 | banana       | 20.660 | apple          | 18.448 |  
| sandwich      | 28.772 | orange       | 28.911 | broccoli       | 21.992 |  
| carrot        | 18.264 | hot dog      | 28.677 | pizza          | 47.652 |  
| donut         | 40.860 | cake         | 30.599 | chair          | 24.919 |  
| couch         | 40.305 | potted plant | 23.166 | bed            | 34.359 |  
| dining table  | 24.119 | toilet       | 56.667 | tv             | 51.158 |  
| laptop        | 52.544 | mouse        | 58.780 | remote         | 25.107 |  
| keyboard      | 47.750 | cell phone   | 30.600 | microwave      | 49.704 |  
| oven          | 27.081 | toaster      | 26.784 | sink           | 32.348 |  
| refrigerator  | 49.154 | book         | 12.779 | clock          | 47.372 |  
| vase          | 32.067 | scissors     | 22.869 | teddy bear     | 39.427 |  
| hair drier    | 5.461  | toothbrush   | 19.293 |                |        |
