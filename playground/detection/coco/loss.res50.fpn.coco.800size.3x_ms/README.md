# loss.res50.fpn.coco.800size.3x_ms  

seed: 51748071

## Evaluation results for bbox:  

```  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.385
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.546
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.424
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.243
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.479
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.566
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.622
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.425
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.654
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.779
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 38.508 | 54.583 | 42.416 | 24.278 | 42.242 | 47.926 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 55.169 | bicycle      | 28.793 | car            | 44.625 |  
| motorcycle    | 41.275 | airplane     | 63.620 | bus            | 62.802 |  
| train         | 57.568 | truck        | 28.984 | boat           | 23.655 |  
| traffic light | 26.291 | fire hydrant | 64.568 | stop sign      | 62.069 |  
| parking meter | 39.397 | bench        | 20.010 | bird           | 34.109 |  
| cat           | 59.474 | dog          | 56.149 | horse          | 54.343 |  
| sheep         | 50.650 | cow          | 57.499 | elephant       | 62.275 |  
| bear          | 68.256 | zebra        | 67.741 | giraffe        | 67.470 |  
| backpack      | 14.588 | umbrella     | 34.578 | handbag        | 13.354 |  
| tie           | 30.790 | suitcase     | 34.793 | frisbee        | 64.653 |  
| skis          | 20.398 | snowboard    | 29.408 | sports ball    | 49.099 |  
| kite          | 42.613 | baseball bat | 22.198 | baseball glove | 33.862 |  
| skateboard    | 48.933 | surfboard    | 29.512 | tennis racket  | 45.521 |  
| bottle        | 36.571 | wine glass   | 33.636 | cup            | 39.598 |  
| fork          | 27.302 | knife        | 16.303 | spoon          | 12.789 |  
| bowl          | 36.349 | banana       | 23.036 | apple          | 18.853 |  
| sandwich      | 30.261 | orange       | 29.867 | broccoli       | 21.440 |  
| carrot        | 20.893 | hot dog      | 31.991 | pizza          | 44.941 |  
| donut         | 46.002 | cake         | 34.270 | chair          | 25.404 |  
| couch         | 40.496 | potted plant | 24.738 | bed            | 37.394 |  
| dining table  | 25.636 | toilet       | 56.370 | tv             | 54.205 |  
| laptop        | 52.698 | mouse        | 59.233 | remote         | 25.636 |  
| keyboard      | 45.584 | cell phone   | 30.076 | microwave      | 53.402 |  
| oven          | 31.793 | toaster      | 28.180 | sink           | 34.512 |  
| refrigerator  | 50.388 | book         | 12.514 | clock          | 49.221 |  
| vase          | 36.323 | scissors     | 22.039 | teddy bear     | 42.655 |  
| hair drier    | 7.286  | toothbrush   | 19.637 |                |        |
