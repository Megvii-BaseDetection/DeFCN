# fcos.res50.fpn.coco.800size.3x_ms.wo_ctrness  

seed: 47789800

## Evaluation results for bbox:  

```  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.602
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.441
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.241
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.452
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.524
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.548
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.382
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.623
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.731
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 40.944 | 60.167 | 44.113 | 24.072 | 45.182 | 52.421 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 54.384 | bicycle      | 31.306 | car            | 44.309 |  
| motorcycle    | 42.440 | airplane     | 67.711 | bus            | 66.455 |  
| train         | 64.182 | truck        | 35.696 | boat           | 26.276 |  
| traffic light | 27.771 | fire hydrant | 67.507 | stop sign      | 65.803 |  
| parking meter | 42.677 | bench        | 22.744 | bird           | 36.077 |  
| cat           | 65.877 | dog          | 61.872 | horse          | 55.550 |  
| sheep         | 53.451 | cow          | 58.303 | elephant       | 65.260 |  
| bear          | 72.269 | zebra        | 69.098 | giraffe        | 66.824 |  
| backpack      | 16.122 | umbrella     | 38.972 | handbag        | 15.176 |  
| tie           | 33.290 | suitcase     | 39.192 | frisbee        | 65.786 |  
| skis          | 21.490 | snowboard    | 35.935 | sports ball    | 47.188 |  
| kite          | 44.031 | baseball bat | 28.208 | baseball glove | 36.072 |  
| skateboard    | 52.649 | surfboard    | 31.155 | tennis racket  | 47.645 |  
| bottle        | 38.152 | wine glass   | 37.058 | cup            | 41.280 |  
| fork          | 33.254 | knife        | 14.530 | spoon          | 14.718 |  
| bowl          | 37.720 | banana       | 23.842 | apple          | 19.079 |  
| sandwich      | 33.285 | orange       | 31.455 | broccoli       | 23.420 |  
| carrot        | 19.758 | hot dog      | 32.770 | pizza          | 51.021 |  
| donut         | 45.210 | cake         | 35.831 | chair          | 27.455 |  
| couch         | 43.319 | potted plant | 27.762 | bed            | 40.798 |  
| dining table  | 26.791 | toilet       | 61.554 | tv             | 55.279 |  
| laptop        | 57.426 | mouse        | 62.401 | remote         | 31.136 |  
| keyboard      | 47.982 | cell phone   | 33.081 | microwave      | 55.147 |  
| oven          | 33.120 | toaster      | 36.481 | sink           | 38.436 |  
| refrigerator  | 53.491 | book         | 12.543 | clock          | 48.801 |  
| vase          | 37.689 | scissors     | 26.647 | teddy bear     | 44.404 |  
| hair drier    | 6.932  | toothbrush   | 17.679 |                |        |
