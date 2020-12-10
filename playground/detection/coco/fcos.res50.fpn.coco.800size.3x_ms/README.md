# fcos.res50.fpn.coco.800size.3x_ms  

seed: 54525629

## Evaluation results for bbox:  

```  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.600
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.442
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.259
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.448
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.523
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.335
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.553
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.413
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.632
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.736
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 41.144 | 59.965 | 44.163 | 25.943 | 44.848 | 52.289 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 55.827 | bicycle      | 31.435 | car            | 45.076 |  
| motorcycle    | 43.151 | airplane     | 66.558 | bus            | 66.629 |  
| train         | 64.856 | truck        | 36.514 | boat           | 26.164 |  
| traffic light | 28.226 | fire hydrant | 67.100 | stop sign      | 63.121 |  
| parking meter | 45.387 | bench        | 22.123 | bird           | 36.271 |  
| cat           | 66.592 | dog          | 61.370 | horse          | 57.585 |  
| sheep         | 52.973 | cow          | 58.292 | elephant       | 65.086 |  
| bear          | 70.675 | zebra        | 69.512 | giraffe        | 66.838 |  
| backpack      | 16.581 | umbrella     | 38.987 | handbag        | 16.546 |  
| tie           | 33.051 | suitcase     | 38.667 | frisbee        | 67.935 |  
| skis          | 21.875 | snowboard    | 35.698 | sports ball    | 48.339 |  
| kite          | 43.205 | baseball bat | 27.600 | baseball glove | 37.712 |  
| skateboard    | 52.775 | surfboard    | 32.483 | tennis racket  | 48.106 |  
| bottle        | 38.856 | wine glass   | 37.560 | cup            | 42.678 |  
| fork          | 31.549 | knife        | 17.275 | spoon          | 16.030 |  
| bowl          | 40.001 | banana       | 23.947 | apple          | 19.621 |  
| sandwich      | 32.222 | orange       | 29.951 | broccoli       | 22.492 |  
| carrot        | 22.061 | hot dog      | 29.331 | pizza          | 50.300 |  
| donut         | 46.797 | cake         | 36.149 | chair          | 28.383 |  
| couch         | 43.713 | potted plant | 26.597 | bed            | 38.193 |  
| dining table  | 26.747 | toilet       | 61.831 | tv             | 55.705 |  
| laptop        | 56.581 | mouse        | 60.873 | remote         | 29.503 |  
| keyboard      | 47.679 | cell phone   | 35.188 | microwave      | 55.802 |  
| oven          | 33.985 | toaster      | 37.350 | sink           | 36.474 |  
| refrigerator  | 53.121 | book         | 13.460 | clock          | 47.878 |  
| vase          | 37.862 | scissors     | 25.866 | teddy bear     | 45.117 |  
| hair drier    | 9.850  | toothbrush   | 19.998 |                |        |
