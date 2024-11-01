import numpy as np 
import cv2 

class Utils : 
    def __init__(self):
        self._object_count = {}
        self.label_count = "'{name}' : {count}"
        self.roi_point = []
        self.roi_point_frame_size = (0,0)
        self.frame_size = (0,0)

    def set_roi(self, roi_point, roi_point_frame_size):
        self.roi_point = roi_point
        self.roi_point_frame_size = roi_point_frame_size

    def normalized_roi_point(self):
        point = []
        for item in self.roi_point :
            point.append([
                int(item[0] * self.frame_size[0] / self.roi_point_frame_size[0]), # for width
                int(item[1] * self.frame_size[1] / self.roi_point_frame_size[1])  # for height
                ])
        return np.array(point) 

    def print_object_count(self):
        for name in self._object_count:
            print(self.label_count.format(name=name, count=self._object_count[name]))

    def reset_counter (self): 
        for name in self._object_count :
            self._object_count[name] = 0

    def count_object_in_roi(self, name, box):
        # increment counter without roi 
        if len(self.roi_point) == 0 :
            self._object_count[name] = self._object_count.get(name, 0) + 1 

        elif len(self.roi_point) > 1 :
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            # cv2.pointPolygonTest function returns +1, -1, or 0 to indicate if a point is inside, outside, or on the contour
            res_1 = cv2.pointPolygonTest(self.normalized_roi_point(), (x,y), False) # top left
            res_2 = cv2.pointPolygonTest(self.normalized_roi_point(), (x+w,y), False) # top right
            res_3 = cv2.pointPolygonTest(self.normalized_roi_point(), (x+w,y+h), False) # bottom right 
            res_4 = cv2.pointPolygonTest(self.normalized_roi_point(), (x,y+h), False) # bottom left

            # resverse check
            box_point = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
            res = -1
            for p in self.normalized_roi_point():
                res = max(res, cv2.pointPolygonTest(box_point, (int(p[0]), int(p[1])), False)) # bottom left

            if (np.max(np.array([res_1, res_2, res_3, res_4, res]))) > -1 : # if at least there is 0 or 1
                self._object_count[name] = self._object_count.get(name, 0) + 1
            elif name not in self._object_count :
                self._object_count[name] = 0

    def draw_object_count(self, img, x0, y0, font_size=0.4, color=(255,127,0), text_color=(255,255,255), padding = 10):
        
        if len(self.roi_point) > 1 :
            img = cv2.polylines(img, [self.normalized_roi_point()], isClosed=True, color=color, thickness=2)

        for name in self._object_count:
            label = self.label_count.format(name=name, count=self._object_count[name])
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
            cv2.rectangle(img,
                (x0, y0 - h - baseline),  
                (x0 + w, y0 + baseline), 
                color, 
                -1)
            cv2.putText(img, 
                        label, 
                        (x0, y0),                   
                        cv2.FONT_HERSHEY_SIMPLEX,     
                        font_size,                          
                        text_color,                
                        1,
                        cv2.LINE_AA) 
            y0 = y0 + h + padding
        
        # clear counter
        self.reset_counter()
        return img

    def draw_ped(self, img, label, x0, y0, xt, yt, font_size=0.4, color=(255,127,0), text_color=(255,255,255)):

        y0, yt = max(y0 - 15, 0) , min(yt + 15, img.shape[0])
        x0, xt = max(x0 - 15, 0) , min(xt + 15, img.shape[1])

        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
        cv2.rectangle(img,
                        (x0, y0 + baseline),  
                        (max(xt, x0 + w), yt), 
                        color, 
                        2)
        cv2.rectangle(img,
                        (x0, y0 - h - baseline),  
                        (x0 + w, y0 + baseline), 
                        color, 
                        -1)
        cv2.rectangle(img,
                        (x0, y0 - h - baseline),  
                        (x0 + w, y0 + baseline), 
                        color, 
                        2)  
        cv2.putText(img, 
                    label, 
                    (x0, y0),                   
                    cv2.FONT_HERSHEY_SIMPLEX,     
                    font_size,                          
                    text_color,                
                    1,
                    cv2.LINE_AA) 
        return img
    
    def postprocess_darknet(self, outs, frame, classes, 
                        confThreshold = 0.5, nmsThreshold = 0.3, font_size=0.8, 
                        color=(255,127,0), text_color=(255,255,255)):

            frame_h, frame_w, ___ = frame.shape
            self.frame_size = frame_h, frame_w

            classIds = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]

                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    c_x = int(detection[0] * frame_w)
                    c_y = int(detection[1] * frame_h)
                    w = int(detection[2] * frame_w)
                    h = int(detection[3] * frame_h)
                    x = int(c_x - w / 2)
                    y = int(c_y - h / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
            for i in indices:
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                label = '%s: %.1f%%' % (classes[classIds[i]], (confidences[i]*100))
                frame = self.draw_ped(frame, label, x, y, x+w, y+h, color=color, text_color=text_color, font_size=font_size)
            
                # calc object counter
                self.count_object_in_roi(classes[class_ids[i]], box)

            return frame

    def postprocess_onnx(self, outs, frame, classes, 
                        confThreshold = 0.5, nmsThreshold = 0.3, font_size=0.8, 
                        color=(255,127,0), text_color=(255,255,255), input_size=[320,320]):

            frame_h, frame_w, ___ = frame.shape
            scale_horizontal = frame_w / input_size[0]
            scale_vertical = frame_h / input_size[1]
            self.frame_size = frame_h, frame_w


            # Prepare output array
            outputs = np.array([cv2.transpose(outs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []

            # Iterate through output to collect bounding boxes, confidence scores, and class IDs
            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (___, maxScore, ____, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)

                c_x = int(outputs[0][i][0] * scale_horizontal)
                c_y = int(outputs[0][i][1] * scale_vertical)
                w = int(outputs[0][i][2] * scale_horizontal)
                h = int(outputs[0][i][3] * scale_vertical)
                x = int(c_x - w / 2)
                y = int(c_y - h / 2)

                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                boxes.append([x, y, w, h])

            # Apply NMS (Non-maximum suppression)
            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold)

            # Iterate through NMS results to draw bounding boxes and labels
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                label = '%s: %.1f%%' % (classes[class_ids[index]], (scores[index]*100))
                frame = self.draw_ped(frame, label, x, y, x+w, y+h, color=color, text_color=text_color, font_size=font_size) 

                # calc object counter
                self.count_object_in_roi(classes[class_ids[index]], box)
            return frame

    def postprocess_onnx_frcnn(self, outs, frame, classes, 
                        confThreshold = 0.5, font_size=0.8, 
                        color=(255,127,0), text_color=(255,255,255), input_size=[224,224]):

        frame_h, frame_w, _ = frame.shape
        scale_horizontal = frame_w / input_size[0]
        scale_vertical = frame_h / input_size[1]

        boxes = outs[0]  # Bounding boxes
        labels = outs[1]  # Class labels
        scores = outs[2]  # Confidence scores

        for box, label, score in zip(boxes, labels, scores):
            if score >= confThreshold :  
                # Extract box coordinates
                x_min, y_min, x_max, y_max = box
                x_min = int(x_min * scale_horizontal)
                x_max = int(x_max * scale_horizontal)
                y_min = int(y_min * scale_vertical)
                y_max = int(y_max * scale_vertical)

                # Draw the bounding box on the image
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color, thickness=2)

                # Prepare label and score text
                score = score * 100
                label_text = f"{classes[label]}: {score:.1f}%" if classes else f"Label {label}: {score:.1f}%"

                # Add label text above the bounding box
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
                cv2.rectangle(frame, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
                cv2.putText(frame, label_text, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, 1)
        
        return frame

    def rescale_box(self, boxes, original_image, input_size=[224,224]):
        h, w, _ = original_image.shape
        scale_hor = w / input_size[0]
        scale_ver = h / input_size[1]
        new_box = []
        for box in boxes :
            x_min, y_min, x_max, y_max = box
            x_min = int(x_min * scale_hor)
            x_max = int(x_max * scale_hor)
            y_min = int(y_min * scale_ver)
            y_max = int(y_max * scale_ver)
            new_box.append([x_min, y_min, x_max, y_max])
        print(new_box)
        return np.array(new_box)