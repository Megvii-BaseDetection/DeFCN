# fcos.res50.fpn.coco.800size.3x_ms.wo_ctrness  

seed: 12139822

## Evaluation results for bbox:  

```  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.407
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.600
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.436
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.242
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.445
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.518
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.332
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.543
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.581
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.393
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.617
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.725
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 40.677 | 60.006 | 43.634 | 24.231 | 44.479 | 51.841 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 54.377 | bicycle      | 30.210 | car            | 44.451 |  
| motorcycle    | 42.588 | airplane     | 66.046 | bus            | 65.609 |  
| train         | 64.081 | truck        | 35.252 | boat           | 26.253 |  
| traffic light | 27.736 | fire hydrant | 68.486 | stop sign      | 67.535 |  
| parking meter | 41.329 | bench        | 22.756 | bird           | 36.113 |  
| cat           | 65.276 | dog          | 60.778 | horse          | 56.514 |  
| sheep         | 52.929 | cow          | 57.701 | elephant       | 64.423 |  
| bear          | 73.913 | zebra        | 67.809 | giraffe        | 66.623 |  
| backpack      | 15.635 | umbrella     | 38.387 | handbag        | 14.926 |  
| tie           | 31.336 | suitcase     | 38.397 | frisbee        | 66.434 |  
| skis          | 22.340 | snowboard    | 28.932 | sports ball    | 47.162 |  
| kite          | 43.187 | baseball bat | 27.201 | baseball glove | 36.671 |  
| skateboard    | 50.435 | surfboard    | 31.592 | tennis racket  | 47.268 |  
| bottle        | 37.715 | wine glass   | 36.252 | cup            | 41.879 |  
| fork          | 32.723 | knife        | 14.435 | spoon          | 13.741 |  
| bowl          | 39.995 | banana       | 23.464 | apple          | 16.328 |  
| sandwich      | 31.708 | orange       | 30.606 | broccoli       | 23.389 |  
| carrot        | 20.206 | hot dog      | 33.945 | pizza          | 51.566 |  
| donut         | 43.895 | cake         | 36.496 | chair          | 26.933 |  
| couch         | 42.589 | potted plant | 26.532 | bed            | 38.327 |  
| dining table  | 26.080 | toilet       | 59.984 | tv             | 54.864 |  
| laptop        | 57.717 | mouse        | 63.120 | remote         | 29.552 |  
| keyboard      | 49.811 | cell phone   | 34.040 | microwave      | 53.641 |  
| oven          | 31.723 | toaster      | 38.291 | sink           | 38.405 |  
| refrigerator  | 53.482 | book         | 12.060 | clock          | 49.026 |  
| vase          | 37.447 | scissors     | 26.607 | teddy bear     | 45.463 |  
| hair drier    | 9.243  | toothbrush   | 22.177 |                |        |
