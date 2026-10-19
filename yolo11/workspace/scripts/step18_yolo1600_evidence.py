"""Record the native validator's existing outputs; no prediction or metric logic changes."""
import json
from pathlib import Path


def make_validator(destination, image_ids, save_json, require):
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class EvidenceValidator(DetectionValidator):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            require(self.save_dir.resolve()==destination.resolve(),'Unexpected native output directory')
            self.exported_rows=[]

        def init_metrics(self,model):
            super().init_metrics(model)
            require(self.names=={0:'broken',1:'loose'} and self.nc==2,'Wrong checkpoint target classes')
            require(not self.args.save_json and not self.is_custom_json and not self.is_coco and not self.is_lvis,
                    'Native AP protocol unexpectedly changed')

        def update_metrics(self,preds,batch):
            super().update_metrics(preds,batch)
            # Export AFTER the original native update. scale_preds clones boxes.
            for index,pred in enumerate(preds):
                prepared=self._prepare_batch(index,batch)
                scaled=self.scale_preds(pred,prepared)
                path=Path(prepared['im_file'])
                boxes=scaled['bboxes'].detach().cpu()
                scores=scaled['conf'].detach().cpu()
                labels=scaled['cls'].detach().cpu().to(torch.int64)
                require(torch.isfinite(boxes).all().item() and torch.isfinite(scores).all().item(),'Non-finite predictions')
                require(set(labels.tolist())<={0,1} and len(scores)<=300,'Prediction class/count changed')
                row=dict(image_id=image_ids[path.name],image_path=str(path),
                    original_shape=list(prepared['ori_shape']),input_shape=list(prepared['imgsz']),
                    ratio_pad=prepared['ratio_pad'],boxes_xyxy=boxes.tolist(),model_labels=labels.tolist(),
                    scores=scores.tolist(),returned_count=len(scores),visible_count=int((scores>.30).sum().item()))
                self.exported_rows.append(row)
                with (destination/'predictions_details.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                print(f'TEST_IMAGE={len(self.exported_rows)}/23 {path.name} returned={len(scores)} visible_above_0.3={row["visible_count"]}',flush=True)

        def get_stats(self):
            # Save exactly the arrays consumed by the original native AP/P/R calculation.
            arrays={key:np.concatenate(values,0) for key,values in self.metrics.stats.items()}
            np.savez_compressed(destination/'native_match_stats.npz',**arrays)
            result=super().get_stats()
            require(self.metrics.nt_per_class.tolist()==[8,28],'Native evaluator test GT counts differ')
            save_json(destination/'actual_evaluation_args.json',vars(self.args))
            save_json(destination/'predictions_details.json',self.exported_rows)
            return result

    return EvidenceValidator


def draw_saved_predictions(destination,rows):
    from PIL import Image,ImageDraw,ImageFont
    output=destination/'predicted_images'
    output.mkdir()
    for row in rows:
        with Image.open(row['image_path']) as source:
            image=source.convert('RGB')
        require_size=tuple(row['original_shape'][:2][::-1])
        if image.size!=require_size:
            raise AssertionError('Original image dimensions differ during drawing')
        draw=ImageDraw.Draw(image)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',max(18,image.width//120))
        for box,label,score in zip(row['boxes_xyxy'],row['model_labels'],row['scores']):
            if score<=.30:continue
            color=((240,60,60),(20,210,230))[label]
            draw.rectangle(box,outline=color,width=max(2,image.width//600))
            draw.text((box[0],max(0,box[1]-24)),f'{("broken","loose")[label]} {score:.3f}',font=font,fill=color)
        image.save(output/(Path(row['image_path']).stem+'.jpg'),quality=95)
