# anchor.res50.fpn.coco.800size.3x_ms  

seed: 8486251

## Evaluation results for bbox:  

```  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.371
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.539
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.407
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.228
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.404
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.468
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.548
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.601
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.401
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.639
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.744
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 37.107 | 53.947 | 40.672 | 22.816 | 40.383 | 46.752 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 52.255 | bicycle      | 26.260 | car            | 41.679 |  
| motorcycle    | 39.210 | airplane     | 61.543 | bus            | 62.401 |  
| train         | 60.828 | truck        | 33.554 | boat           | 20.856 |  
| traffic light | 24.170 | fire hydrant | 63.962 | stop sign      | 63.931 |  
| parking meter | 40.240 | bench        | 19.549 | bird           | 31.676 |  
| cat           | 61.107 | dog          | 53.378 | horse          | 52.635 |  
| sheep         | 47.743 | cow          | 53.822 | elephant       | 58.579 |  
| bear          | 64.377 | zebra        | 65.260 | giraffe        | 64.529 |  
| backpack      | 13.192 | umbrella     | 34.509 | handbag        | 12.622 |  
| tie           | 28.685 | suitcase     | 36.227 | frisbee        | 60.704 |  
| skis          | 18.887 | snowboard    | 25.908 | sports ball    | 46.132 |  
| kite          | 42.299 | baseball bat | 22.943 | baseball glove | 32.847 |  
| skateboard    | 45.960 | surfboard    | 29.553 | tennis racket  | 41.965 |  
| bottle        | 34.789 | wine glass   | 31.639 | cup            | 37.806 |  
| fork          | 23.313 | knife        | 14.087 | spoon          | 11.881 |  
| bowl          | 36.950 | banana       | 20.639 | apple          | 17.708 |  
| sandwich      | 29.663 | orange       | 28.906 | broccoli       | 20.481 |  
| carrot        | 17.646 | hot dog      | 27.485 | pizza          | 47.625 |  
| donut         | 40.716 | cake         | 32.651 | chair          | 24.224 |  
| couch         | 39.296 | potted plant | 23.100 | bed            | 35.414 |  
| dining table  | 24.427 | toilet       | 55.314 | tv             | 51.653 |  
| laptop        | 53.287 | mouse        | 56.924 | remote         | 24.388 |  
| keyboard      | 47.950 | cell phone   | 31.175 | microwave      | 53.721 |  
| oven          | 28.448 | toaster      | 28.450 | sink           | 31.410 |  
| refrigerator  | 50.166 | book         | 12.128 | clock          | 47.060 |  
| vase          | 32.070 | scissors     | 25.277 | teddy bear     | 39.712 |  
| hair drier    | 8.009  | toothbrush   | 15.033 |                |        |
