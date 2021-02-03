# fcos.res50.fpn.coco.800size.3x_ms  

seed: 9476764

## Evaluation results for bbox:  

```  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.414
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.601
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.449
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.256
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.449
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.335
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.553
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.591
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.635
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.735
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 41.393 | 60.086 | 44.923 | 25.561 | 44.897 | 53.084 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 55.999 | bicycle      | 32.520 | car            | 45.318 |  
| motorcycle    | 43.277 | airplane     | 67.218 | bus            | 66.594 |  
| train         | 63.735 | truck        | 37.657 | boat           | 24.362 |  
| traffic light | 27.385 | fire hydrant | 67.430 | stop sign      | 63.445 |  
| parking meter | 43.762 | bench        | 22.987 | bird           | 36.695 |  
| cat           | 67.516 | dog          | 62.411 | horse          | 56.741 |  
| sheep         | 53.373 | cow          | 58.669 | elephant       | 64.608 |  
| bear          | 71.341 | zebra        | 69.199 | giraffe        | 68.521 |  
| backpack      | 16.543 | umbrella     | 38.757 | handbag        | 15.861 |  
| tie           | 32.415 | suitcase     | 39.008 | frisbee        | 68.187 |  
| skis          | 20.592 | snowboard    | 32.193 | sports ball    | 47.290 |  
| kite          | 42.626 | baseball bat | 28.741 | baseball glove | 36.490 |  
| skateboard    | 54.258 | surfboard    | 33.234 | tennis racket  | 49.328 |  
| bottle        | 39.079 | wine glass   | 37.518 | cup            | 42.291 |  
| fork          | 31.993 | knife        | 18.649 | spoon          | 15.694 |  
| bowl          | 41.004 | banana       | 24.253 | apple          | 19.303 |  
| sandwich      | 31.717 | orange       | 31.743 | broccoli       | 23.667 |  
| carrot        | 21.484 | hot dog      | 31.344 | pizza          | 52.775 |  
| donut         | 46.693 | cake         | 37.320 | chair          | 28.833 |  
| couch         | 44.514 | potted plant | 28.510 | bed            | 38.643 |  
| dining table  | 26.747 | toilet       | 59.289 | tv             | 55.466 |  
| laptop        | 57.641 | mouse        | 62.759 | remote         | 31.570 |  
| keyboard      | 47.522 | cell phone   | 35.813 | microwave      | 52.229 |  
| oven          | 32.445 | toaster      | 41.552 | sink           | 36.470 |  
| refrigerator  | 53.942 | book         | 13.845 | clock          | 48.035 |  
| vase          | 36.108 | scissors     | 26.815 | teddy bear     | 47.294 |  
| hair drier    | 13.241 | toothbrush   | 19.316 |                |        |
