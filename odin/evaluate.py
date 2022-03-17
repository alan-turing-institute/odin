# Add methods to evaluate the performance: map, joel's method
import numpy as np


# retrieve mAP from the bboxes

# get list of predicted boxes for an images and a list of truth boxes
# if there's only a single class then you can
# arrange predcited boxes according to 'objectness' ie: class score
# grab the first predicted box and determine IOU of with all the truth boxes for that image
# if any IOU>0, assign it to the truth box with greatest IOU; else assign it as a FP
# remove predicted box from list; if it was assigned to a truth box, also remove that from the list
# repeat util you run out of predicted boxes
# any remaining truth boxes can be assigned as false negatives

def iou_bbox(gt_box, pred_box):
    """
    Method to calculate the Intersection over Union between two boxes
    """
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0] + gt_box[2], pred_box[0] + pred_box[2]),
                              min(gt_box[1] + gt_box[3], pred_box[1] + pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection

    iou = intersection / union
    return iou


def all_iou(box, list_gt):
    """
    Given a predicted box, retrieve the IoU with all the ground truth bboxes
    Input
    box - the predicted bbox
    list_gt - list of all the ground truth bboxes
    """
    result = [iou_bbox(box, gt) for gt in list_gt]
    return result


def confusion_matrix(ranked_predictions, gt, threshold=0):
    """
    For each box in the list of predicted + gt boxes, attribute TP, FP, FN.
    """

    TP = 0
    FP = 0
    # Get the highest score, calculate all IoU of that score with the ground-truths

    while len(ranked_predictions) > 0:
        scores = all_iou(ranked_predictions[-1], gt)
        max_iou = max(scores)
        if max_iou > threshold:
            TP += 1
            max_ind = scores.index(max_iou)
            gt.pop(max_ind)
        else:
            FP += 0
        ranked_predictions.pop()

    FN = len(gt)

    return TP, FN, FP


def evaluate_image(gt_dict, pred_dict):
    """
    Method to evaluate the precision, recall and f1 of the model for bounding boxes in one image

    """

    # Step 1: organise all scores, sort predictions per score
    predictions = scored_bbox(pred_dict)
    ranked_predictions = predictions[predictions[:, -1].argsort()]
    gt = gt_dict['boxes']

    TP, FN, FP = confusion_matrix(ranked_predictions, gt)
    # Precision

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_one = (2 * precision * recall) / (precision + recall)

    return precision, recall, f_one


def evaluate_all(ground_truth, predictions):
    """
    Method to evaluate the precision, recall and f1 for all the images
    """


def mean_average_precision(precisions, recalls):
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])


def scored_bbox(input_pred):
    """
    Convert the predicted bbox into the format [xmin,ymin,xmax,ymax,label,score] to calculate mAP
    Input:
    pred_bbox: {boxes, labels, scores}

    Output:
    predictions: [xmin, ymin, xmax, ymax, class_id, confidence]

    """
    n_pred_boxes = len(input_pred['scores'])
    predictions = np.array([n_pred_boxes, 6])

    for i in range(0, n_pred_boxes):
        predictions[i, 0] = input_pred['boxes'][i][0]
        predictions[i, 1] = input_pred['boxes'][i][1]
        predictions[i, 2] = input_pred['boxes'][i][2]
        predictions[i, 3] = input_pred['boxes'][i][3]
        predictions[i, 4] = input_pred['labels'][i]
        predictions[i, 5] = input_pred['scores'][i]
    return predictions
