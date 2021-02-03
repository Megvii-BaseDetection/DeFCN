# loss.res50.fpn.coco.800size.3x_ms  

seed: 3751988

## Evaluation results for bbox:  

```  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.549
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.427
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.238
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.327
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.565
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.622
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.419
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.656
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.788
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 38.708 | 54.872 | 42.708 | 23.793 | 42.364 | 48.888 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 55.156 | bicycle      | 28.500 | car            | 44.210 |  
| motorcycle    | 40.619 | airplane     | 64.635 | bus            | 62.010 |  
| train         | 60.690 | truck        | 31.329 | boat           | 24.446 |  
| traffic light | 28.411 | fire hydrant | 63.136 | stop sign      | 63.268 |  
| parking meter | 39.984 | bench        | 22.074 | bird           | 34.874 |  
| cat           | 61.970 | dog          | 57.768 | horse          | 55.772 |  
| sheep         | 51.466 | cow          | 57.623 | elephant       | 62.707 |  
| bear          | 66.627 | zebra        | 67.822 | giraffe        | 67.217 |  
| backpack      | 13.496 | umbrella     | 37.257 | handbag        | 13.215 |  
| tie           | 30.037 | suitcase     | 35.837 | frisbee        | 63.655 |  
| skis          | 20.689 | snowboard    | 26.305 | sports ball    | 48.801 |  
| kite          | 42.445 | baseball bat | 22.402 | baseball glove | 33.640 |  
| skateboard    | 48.489 | surfboard    | 30.267 | tennis racket  | 45.932 |  
| bottle        | 37.132 | wine glass   | 34.082 | cup            | 39.278 |  
| fork          | 26.000 | knife        | 14.181 | spoon          | 14.024 |  
| bowl          | 37.208 | banana       | 23.155 | apple          | 18.371 |  
| sandwich      | 31.738 | orange       | 30.707 | broccoli       | 23.113 |  
| carrot        | 20.558 | hot dog      | 31.242 | pizza          | 46.054 |  
| donut         | 45.652 | cake         | 34.416 | chair          | 25.191 |  
| couch         | 39.924 | potted plant | 24.988 | bed            | 36.558 |  
| dining table  | 26.308 | toilet       | 56.805 | tv             | 53.605 |  
| laptop        | 51.866 | mouse        | 58.877 | remote         | 25.243 |  
| keyboard      | 48.115 | cell phone   | 30.290 | microwave      | 55.947 |  
| oven          | 31.628 | toaster      | 28.743 | sink           | 33.973 |  
| refrigerator  | 48.122 | book         | 12.652 | clock          | 47.065 |  
| vase          | 35.688 | scissors     | 25.522 | teddy bear     | 42.451 |  
| hair drier    | 8.780  | toothbrush   | 16.637 |                |        |
