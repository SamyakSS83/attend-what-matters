import os, math, copy
import numpy as np 
import pandas as pd
import pydicom
import random
import pickle
import hashlib

from tqdm.auto import tqdm

import torchvision.transforms.v2 as v2
import torch

from image_preprocessing.process_dicom import convert_to_numpy, convert_16bit_image_to_8bit
from image_preprocessing.crop import get_cropping_coordinates, crop_image, pad_crop_image

DISJOINT = False
FOCAL = False


class all_mammo():
    def __init__(self, csv_path, img_base, text_base, iou_threshold=0.5, topk=5, img_size=448, enable_augmentation=False, cache_dir="./cache/"):

        self.img_base = img_base
        self.text_base = text_base
        self.cache_dir = cache_dir

        self.img_size = img_size
        self.image_path_list, self.label = self.csv_to_list(csv_path)
        self.topk = topk
        self.iou_threshold = iou_threshold
        self.all_proposals = self.get_all_proposals()
        self.enable_augmentation = enable_augmentation


    def __len__(self):

        return len(self.label)


    def __getitem__(self, index):

        label = torch.tensor([self.label[index]])
        image_path = self.image_path_list[index]
        image = self.get_image(image_path)
        proposals = self.all_proposals[index]
        box_positions = self.normalize_boxes(proposals, image)
        box_encodings = self.gen_sineembed_for_position(box_positions)
        crops = self.create_crops(image, proposals, self.img_size)
        
        return torch.stack(crops), box_encodings, label

    
    def csv_to_list(self, csv_path):

        df = pd.read_csv(csv_path)

        return (df['im_path'].tolist(), df['cancer'].tolist())

    
    def get_image(self, img_path):

        dicom_image = pydicom.dcmread(os.path.join(self.img_base, img_path))
        pixel_array = convert_to_numpy(dicom_image)
        pixel_array_8bit = convert_16bit_image_to_8bit(pixel_array)
        box_coordinates = get_cropping_coordinates(pixel_array_8bit, padding=15)
        cropped_image_16bit = crop_image(pixel_array, box_coordinates)

        return cropped_image_16bit
    
    def get_breast_contour_bbox(self, image):
        """Extract bounding box of full breast region using contour detection.
        
        Returns bbox in format: [cx, cy, w, h, score=1.0]
        """
        import cv2
        
        # Convert to 8-bit for contour detection
        img_8bit = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        
        # Threshold to get breast tissue (non-black regions)
        _, binary = cv2.threshold(img_8bit, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # Fallback: use entire image
            H, W = image.shape
            return np.array([W/2, H/2, W, H, 1.0], dtype=np.float32)
        
        # Get largest contour (breast region)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Convert to center format with score 1.0
        cx = x + w / 2.0
        cy = y + h / 2.0
        
        return np.array([cx, cy, w, h, 1.0], dtype=np.float32)


    def _get_cache_path(self):
        """Generate unique cache filename based on dataset configuration."""
        # Create hash from config parameters
        config_str = f"{self.text_base}_{self.topk}_{self.iou_threshold}_{FOCAL}_{DISJOINT}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        cache_filename = f"proposals_{config_hash}.pkl"
        return os.path.join(self.cache_dir, cache_filename)
    
    def get_all_proposals(self):
        """Get all proposals with caching support."""
        
        cache_path = self._get_cache_path()
        
        # # Try to load from cache
        if os.path.exists(cache_path):
            print(f"Loading proposals from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify cache is valid (same number of images)
                if len(cached_data) == len(self.image_path_list):
                    print(f"Successfully loaded {len(cached_data)} proposals from cache!")
                    return cached_data
                else:
                    print(f"Cache size mismatch: {len(cached_data)} vs {len(self.image_path_list)}. Regenerating...")
            except Exception as e:
                print(f"Error loading cache: {e}. Regenerating...")
        
        # Generate proposals if cache doesn't exist or is invalid
        print("Generating proposals (this may take a while)...")
        all_proposals = []

        for index, img_path in enumerate(tqdm(self.image_path_list, total=len(self.image_path_list), desc='generating proposals', position=0, leave=True)):
            box_filename = self.image_path_list[index].replace(".dicom","") + '_preds.txt'
            proposal_path = os.path.join(self.text_base, box_filename)
            # handle missing prediction files gracefully
            if not os.path.isfile(proposal_path):
                print(f"Warning: prediction file not found: {proposal_path}. Using fallback proposals.")
                boxes = np.zeros((0, 5), dtype=np.float32)
                proposal_missing = True
            else:
                boxes = np.loadtxt(proposal_path, dtype=np.float32)
                proposal_missing = False

            if FOCAL:
                # FOCAL mode: create disjoint ROIs, then add breast bbox as 0th ROI
                # ensure we have the image to compute breast bbox/fallbacks
                image = self.get_image(self.image_path_list[index])
                breast_bbox = self.get_breast_contour_bbox(image)
                # if there were no boxes, create_proposals will return empty; handle that
                disjoint_rois = self.create_proposals(boxes) if boxes.size else np.zeros((0, 5), dtype=np.float32)
                # Insert breast bbox at position 0, keep topk-1 disjoint ROIs
                proposals = np.vstack([breast_bbox, disjoint_rois[:self.topk-1]])
                # pad to topk with breast_bbox if needed
                if proposals.shape[0] < self.topk:
                    needed = self.topk - proposals.shape[0]
                    pad = np.repeat(breast_bbox[np.newaxis, :], needed, axis=0)
                    proposals = np.vstack([proposals, pad])
            else:
                proposals = self.create_proposals(boxes)
                # pad if not enough proposals
                if proposals.shape[0] < self.topk:
                    if proposals.shape[0] > 0:
                        filler = proposals[0]
                    else:
                        image = self.get_image(self.image_path_list[index])
                        H, W = image.shape
                        filler = np.array([W/2, H/2, W, H, 0.0], dtype=np.float32)
                    need = self.topk - proposals.shape[0]
                    pad = np.repeat(filler[np.newaxis, :], need, axis=0)
                    proposals = np.vstack([proposals, pad])

            # Debug print for the 10th image (index 10), without double-appending
            # if index == 187:
            #     print("Length of boxes for 10th image:", len(boxes))
            #     print("Proposals for 10th image:")
            #     print(proposals)
            #     print("Pairwise IOU matrix (excluding anchor 0 as requested):")
            #     for i in range(1, len(proposals)):
            #         for j in range(i + 1, len(proposals)):
            #             iou = self.calculate_iou(proposals[i], proposals[j])
            #             print(f"IOU between proposal {i} and {j}: {iou}")

            #     # Save a dummy image with all proposals drawn
            #     import matplotlib.pyplot as plt
            #     import matplotlib.patches as patches

            #     # Load the image
            #     img_path = self.image_path_list[index]
            #     dicom_image = pydicom.dcmread(os.path.join(self.img_base, img_path))
            #     pixel_array = convert_to_numpy(dicom_image)
            #     pixel_array_8bit = convert_16bit_image_to_8bit(pixel_array)
            #     box_coordinates = get_cropping_coordinates(pixel_array_8bit, padding=15)
            #     cropped_image_16bit = crop_image(pixel_array, box_coordinates)

            #     fig, ax = plt.subplots(1)
            #     ax.imshow(cropped_image_16bit, cmap='gray')

            #     for box in proposals:
            #         cx, cy, w, h, *_ = box
            #         x = cx - w / 2
            #         y = cy - h / 2
            #         rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            #         ax.add_patch(rect)

            #     plt.axis('off')
            #     plt.savefig("dummy.png", bbox_inches='tight', pad_inches=0)
            #     plt.close(fig)

            all_proposals.append(proposals)
        
        # Save to cache
        print(f"Saving proposals to cache: {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(all_proposals, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Successfully cached {len(all_proposals)} proposals!")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

        return all_proposals
    
    def create_proposals(self, boxes):

        proposals = self.non_max_suppression(boxes)
        if len(proposals) < self.topk:
            # Randomly sample from the existing proposals and append to the end
            additional_proposals = random.choices(proposals, k=self.topk - len(proposals))
            proposals = np.concatenate([proposals, additional_proposals])
        proposals = proposals[:self.topk]
        
        return proposals
    
    def create_disjoint_proposals(self, boxes, iou_thresh):
        """
        Create proposals ensuring low pairwise IoU among non-anchor ROIs.
        Always keep the highest-score box as anchor at index 0 and exclude it
        from all IoU calculations, as requested.
        """
        boxes = np.asarray(boxes, dtype=np.float32)
        N = len(boxes)
        if N == 0:
            return np.zeros((0, 5), dtype=np.float32)

        # Extract scores (last column if available)
        scores = boxes[:, 4] if boxes.shape[1] >= 5 else np.zeros(N, dtype=np.float32)

        # Sort by score descending and pick the highest as anchor
        order = np.argsort(scores)[::-1]
        anchor_idx = int(order[0])
        selected = [anchor_idx]  # anchor at position 0

        # Greedily pick boxes with IoU <= iou_thresh among already selected NON-ANCHOR boxes
        for idx in order[1:]:
            if len(selected) >= self.topk:
                break
            cand = boxes[idx]
            too_close = False
            for j in selected[1:]:  # ignore anchor for IoU checks
                if self.calculate_iou(cand, boxes[j]) > iou_thresh:
                    too_close = True
                    break
            if not too_close:
                selected.append(int(idx))

        # If still not enough, try to add more distinct boxes from remaining order
        if len(selected) < self.topk:
            for idx in order[1:]:
                if len(selected) >= self.topk:
                    break
                if int(idx) in selected:
                    continue
                cand = boxes[idx]
                too_close = False
                for j in selected[1:]:
                    if self.calculate_iou(cand, boxes[j]) > iou_thresh:
                        too_close = True
                        break
                if not too_close:
                    selected.append(int(idx))

        # If still short, pad by sampling from already selected non-anchor if possible
        if len(selected) < self.topk:
            pool = selected[1:] if len(selected) > 1 else [anchor_idx]
            need = self.topk - len(selected)
            extra = np.random.choice(pool, size=need, replace=True)
            selected.extend([int(x) for x in extra.tolist()])

        proposals = boxes[selected[:self.topk]]
        return proposals

    
    def create_crops(self, image, proposals, img_size):


        ### Get topk augmented or non augmented crops from image

        augmented_transform = v2.Compose([
            v2.Lambda(lambda x: x.repeat(3, 1, 1)),
            v2.Resize(size=(img_size, img_size), antialias=True),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform = v2.Compose([
            v2.Lambda(lambda x: x.repeat(3, 1, 1)),
            v2.Resize(size=(img_size, img_size), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # crop the images for the top k boxes 
        crop_lis = []

        for j,box in enumerate(proposals):
            processed_box = self.box_processing(box[:4], image)
            roi = pad_crop_image(image, processed_box)
            roi_float32 = roi.astype(np.float32) / 65535.0
            roi_tensor = torch.from_numpy(roi_float32).unsqueeze(0)
            if(self.enable_augmentation):
                roi_tensor = augmented_transform(roi_tensor)
            else:
                roi_tensor = transform(roi_tensor)
            crop_lis.append(roi_tensor)
        
        return crop_lis


    def box_processing(self, box, image):

        ### Returns a square bounding box with 20 pixel padding on all four sides

        H, W = image.shape
        cx, cy, w, h = box
        side = max(w, h)
        x1 = int(cx - side/2 - 20)
        y1 = int(cy - side/2 - 20)

        return x1, y1, int(side+40), int(side+40)


    def convert_yolo_pascal(self, box, image):
        
        cx, cy, w, h = box
        x1 = int((cx-w/2)); x2 = int((cx+w/2))
        y1 = int((cy-h/2)); y2 = int((cy+h/2))

        height = abs(x1 - x2)
        width = abs(y1 - y2)
        padding = abs(height - width) // 2

        if(height < width):
            x1 = x1 - padding
            x2 = x2 + padding
        elif(height > width):
            y1 = y1 - padding
            y2 = y2 + padding
        
        x1 = x1 - 20; y1 = y1 - 20
        x2 = x2 + 20; y2 = y2 + 20

        bbox = [x1, y1, x2, y2]

        return bbox

    
    def non_max_suppression(self, boxes):

        ### Remove box if it has more than threshold IOU area with a selected box

        boxes_copy = copy.deepcopy(boxes)
        selected_indices = []
        removed_indices = set()

        for i in range(len(boxes_copy)):
            flag = True
            for idx in selected_indices:
                if(self.calculate_iou(boxes_copy[i], boxes_copy[idx]) > self.iou_threshold):
                    flag = False
                    break
            if(flag):
                selected_indices.append(i)

            if(len(selected_indices) >= self.topk):
                break

        boxes_copy = boxes_copy[selected_indices]

        return boxes_copy


    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1, c1 = box1
        x2, y2, w2, h2, c2 = box2
        
        # Convert to absolute coordinates
        x1 = x1 - w1/2; y1 = y1 - h1/2
        x2 = x2 - w2/2; y2 = y2 - h2/2
        
        # Calculate intersection coordinates
        x_intersection = max(x1, x2)
        y_intersection = max(y1, y2)
        w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
        h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)
        
        # Calculate intersection area
        intersection_area = w_intersection * h_intersection
        
        # Calculate areas of the bounding boxes
        area1 = w1 * h1
        area2 = w2 * h2
        denom = float(area1 + area2 - intersection_area)
        if denom <= 0:
            return 0.0
        iou = intersection_area / denom
        # Clamp to [0, 1] to avoid numerical spillover above 1.0
        if iou < 0.0:
            iou = 0.0
        elif iou > 1.0:
            iou = 1.0
        return float(iou)


    def normalize_boxes(self, boxes, img):

        H, W = img.shape
        normalized_boxes = []
        for box in boxes:
            cx, cy, w, h, _ = box
            normalized_boxes.append([cx / W, cy / H, max(w, h) / W, max(w, h) / H])

        return torch.tensor(normalized_boxes)
        

    def gen_sineembed_for_position(self, pos_tensor, d_model=128):

        ### code/functionality taken from DAB-DETR
        
        scale = 2 * math.pi
        dim_t = torch.arange(d_model // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))
        x_embed = pos_tensor[:, 0] * scale
        y_embed = pos_tensor[:, 1] * scale
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, 2] * scale
            pos_w = w_embed[:, None] / dim_t
            pos_w = torch.stack((pos_w[:, 0::2].sin(), pos_w[:, 1::2].cos()), dim=2).flatten(1)

            h_embed = pos_tensor[:, 3] * scale
            pos_h = h_embed[:, None] / dim_t
            pos_h = torch.stack((pos_h[:, 0::2].sin(), pos_h[:, 1::2].cos()), dim=2).flatten(1)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=1).to(torch.float32)
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))

        return pos


if __name__ == '__main__':

    DATA_CSV = "/home/samyak/scratch/my_vindr/model_specific_preprocessed_data/mmbcd_csvs/train.csv"
    DATA_IMG_BASE = "/home/samyak/scratch/my_vindr/images/"
    DATA_TEXT_BASE = "/home/samyak/scratch/focal_rois/"

    data = all_mammo(DATA_CSV, DATA_IMG_BASE, DATA_TEXT_BASE, topk = 10, enable_augmentation=True)

    X1, X2, y = data[10]
    print(X1.shape)
    print(X2.shape)
    print(X2.dtype)
    print(X2[0])

    print(y)
    
    # Visualization for FOCAL mode: show random image with topk disjoint bboxes
    # if FOCAL:
    #     import matplotlib.pyplot as plt
    #     import matplotlib.patches as patches
    #     import random
        
    #     print("\n" + "="*60)
    #     print("FOCAL MODE VISUALIZATION")
    #     print("="*60)
        
    #     # Pick a random image
    #     random_idx = 0
    #     print(f"\nVisualizing image at index: {random_idx}")
        
    #     # Get image and proposals
    #     image_path = data.image_path_list[random_idx]
    #     image = data.get_image(image_path)
    #     box_filename = image_path.replace(".dicom", "") + '_preds.txt'
    #     boxes = np.loadtxt(os.path.join(data.text_base, box_filename))
    #     if boxes.ndim == 1:
    #         boxes = boxes[np.newaxis, :]
        
    #     # Get breast bbox and disjoint proposals
    #     breast_bbox = data.get_breast_contour_bbox(image)
    #     disjoint_rois = data.create_disjoint_proposals(boxes, 0.3)
    #     all_proposals = np.vstack([breast_bbox, disjoint_rois[:data.topk-1]])
        
    #     print(f"Total ROIs: {len(all_proposals)}")
    #     print(f"0th ROI (breast bbox): cx={breast_bbox[0]:.1f}, cy={breast_bbox[1]:.1f}, w={breast_bbox[2]:.1f}, h={breast_bbox[3]:.1f}")
    #     print(f"Remaining disjoint ROIs: {len(all_proposals)-1}")
        
    #     # Visualize
    #     fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    #     ax.imshow(image, cmap='gray')
        
    #     # Draw each bbox
    #     for i, bbox in enumerate(all_proposals):
    #         cx, cy, w, h, score = bbox
    #         x1 = cx - w/2
    #         y1 = cy - h/2
            
    #         if i == 0:
    #             # 0th ROI: breast bbox in red with thick border
    #             color = 'red'
    #             linewidth = 3
    #             label = f'0: Breast (full)'
    #         else:
    #             # Other ROIs in green
    #             color = 'green'
    #             linewidth = 2
    #             label = f'{i}: score={score:.3f}'
            
    #         rect = patches.Rectangle((x1, y1), w, h, linewidth=linewidth, 
    #                                  edgecolor=color, facecolor='none', label=label)
    #         ax.add_patch(rect)
            
    #         # Add text label
    #         ax.text(x1, y1 - 5, label, color=color, fontsize=10, 
    #                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
    #     ax.set_title(f'FOCAL Mode: Image {random_idx} with {len(all_proposals)} ROIs\n(Red=Breast, Green=Disjoint Proposals)')
    #     ax.axis('off')
    #     plt.tight_layout()
    #     plt.savefig('focal_mode_visualization.png', dpi=150, bbox_inches='tight')
    #     print(f"\nVisualization saved to: focal_mode_visualization.png")
    #     plt.show()


