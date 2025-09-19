import cv2
import matplotlib.pyplot as plt

def detect_boxes(img, show_image=False):
    """
    Detect bounding boxes for symbols in the image.

    Args:
        img (numpy.ndarray): Input image (BGR).
        show_image (bool): Whether to display the image after drawing boxes using Matplotlib.

    Returns:
        tuple: (img_with_boxes, boxes)
            - img_with_boxes: A copy of the original image with bounding boxes drawn.
            - boxes: A list of dicts, each with:
                {
                  'coords': (x, y, w, h),
                  'region': The cropped binary region of the symbol
                }
    """
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold via Otsu (invert so the symbols become white on black)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left-to-right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Calculate minimum area based on image size
    image_height, image_width = img.shape[:2]
    min_area = (image_height * image_width) * 0.0005  # 0.05% of image area
    min_dimension = 5  # Minimum width/height for a symbol

    # Prepare output
    img_with_boxes = img.copy()
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h) if h else 0  # Avoid division by zero

        # Validation check
        is_valid = (
            (area > min_area or (w > min_dimension and h > 0)) and
            (
                (0.1 <= aspect_ratio <= 20) or
                (w > image_width * 0.1 and h >= 1)
            )
        )

        if is_valid:
            # Draw a red rectangle around the detected symbol
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 1)

            # Store the coordinates and symbol region
            symbol_region = binary[y : y + h, x : x + w]
            boxes.append({
                'coords': (x, y, w, h),
                'region': symbol_region
            })

    # Optionally display the image via Matplotlib
    if show_image:
        # Convert BGR to RGB before showing
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Image with Boxes")
        plt.axis("off")
        plt.show()

    return img_with_boxes, boxes

def detect_fraction_bars(img, boxes, show_image=False):
    """
    Detect fraction bars based on bounding boxes.

    A box is considered a fraction bar if:
    1) width > height.
    2) It has at least one "numerator" box (center Y above fraction bar's center),
       and at least one "denominator" box (center Y below fraction bar's center).
    3) Each of those boxes must have a center X that lies horizontally within
       the fraction bar's bounds.

    Args:
        img (numpy.ndarray): The original image in BGR format.
        boxes (list): A list of dictionaries, each having:
            {
                'coords': (x, y, w, h),
                'region': (binary cropped region)
            }
        show_image (bool): Whether to display the resulting image with fraction bars highlighted.

    Returns:
        tuple: (img_with_fraction_bars, fraction_bar_boxes)
            - img_with_fraction_bars: Copy of original image with fraction bars drawn in green (1px line).
            - fraction_bar_boxes: List of box dictionaries identified as fraction bars.
    """
    img_with_fraction_bars = img.copy()
    fraction_bar_boxes = []

    # Precompute the centers for each box to avoid recomputing
    for b in boxes:
        x, y, w, h = b['coords']
        b['center_x'] = x + w / 2.0
        b['center_y'] = y + h / 2.0

    for bar_box in boxes:
        x, y, w, h = bar_box['coords']

        # Condition 1: Potential fraction bar if width > height
        if w > h:
            bar_center_y = bar_box['center_y']
            bar_left = x
            bar_right = x + w

            found_numerator = False
            found_denominator = False

            # Check other boxes for numerator/denominator
            for other_box in boxes:
                # Skip checking itself
                if other_box is bar_box:
                    continue

                other_center_x = other_box['center_x']
                other_center_y = other_box['center_y']

                # Must be horizontally within fraction bar's bounding box
                if bar_left <= other_center_x <= bar_right:
                    # Numerator candidate: center above fraction bar center
                    if other_center_y < bar_center_y:
                        found_numerator = True
                    # Denominator candidate: center below fraction bar center
                    elif other_center_y > bar_center_y:
                        found_denominator = True

                # Early break if both found
                if found_numerator and found_denominator:
                    break

            # If both found, it's a fraction bar
            if found_numerator and found_denominator:
                fraction_bar_boxes.append(bar_box)
                # Draw a green rectangle around the fraction bar (1px thickness)
                cv2.rectangle(
                    img_with_fraction_bars,
                    (x, y), (x + w, y + h),
                    (0, 255, 0), 1
                )

    # Optionally display the image
    if show_image:
        img_rgb = cv2.cvtColor(img_with_fraction_bars, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title("Detected Fraction Bars")
        plt.axis("off")
        plt.show()

    return img_with_fraction_bars, fraction_bar_boxes


def merge_boxes(img, boxes, show_image=False):
    """
    Merge boxes that overlap horizontally and are vertically separated.

    Args:
        img (numpy.ndarray): The original image.
        boxes (list): List of dictionaries containing box coordinates and regions.
        show_image (bool): Whether or not to display the image with merged boxes.

    Returns:
        tuple: (merged_img, merged_boxes)
            - merged_img (numpy.ndarray): The image with merged boxes drawn in blue.
            - merged_boxes (list): List of merged box dictionaries.
    """
    # Create copies so the original data remains unchanged
    merged_img = img.copy()
    merged_boxes = boxes.copy()

    changes_made = True
    merge_count = 0

    # Perform merging logic
    while changes_made:
        changes_made = False
        i = 0
        while i < len(merged_boxes) - 1:
            x1, y1, w1, h1 = merged_boxes[i]['coords']
            center_x1 = x1 + w1 / 2
            center_y1 = y1 + h1 / 2

            j = i + 1
            while j < len(merged_boxes):
                x2, y2, w2, h2 = merged_boxes[j]['coords']
                center_x2 = x2 + w2 / 2
                center_y2 = y2 + h2 / 2

                # Check horizontal overlap by comparing center_x positions
                box1_contains_2 = (x1 <= center_x2 <= x1 + w1)
                box2_contains_1 = (x2 <= center_x1 <= x2 + w2)

                if box1_contains_2 or box2_contains_1:
                    # Boxes overlap horizontally, check vertical positioning
                    if center_y1 != center_y2:
                        # Merge boxes
                        new_x = min(x1, x2)
                        new_y = min(y1, y2)
                        new_w = max(x1 + w1, x2 + w2) - new_x
                        new_h = max(y1 + h1, y2 + h2) - new_y

                        merged_box = {
                            'coords': (new_x, new_y, new_w, new_h),
                            'region': None  # Optionally merge 'region' here
                        }

                        merged_boxes.pop(j)
                        merged_boxes.pop(i)
                        merged_boxes.insert(i, merged_box)

                        merge_count += 1
                        changes_made = True
                        break
                j += 1

            if changes_made:
                break
            i += 1

    # Draw final merged boxes in blue (with 1px thickness)
    for box in merged_boxes:
        x, y, w, h = box['coords']
        cv2.rectangle(merged_img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Optionally display the merged result
    if show_image:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Merged Boxes (Total merges: {merge_count})')
        plt.axis('off')
        plt.show()

    return merged_img, merged_boxes


def extract_box_images(img, boxes, show_images=False):
    """
    Extracts individual (cropped) images from the original image
    based on the bounding boxes.

    Args:
        img (numpy.ndarray): The original image (BGR).
        boxes (list): List of dictionaries, each with a 'coords' key.
                      Example: {'coords': (x, y, w, h), ...}
        show_images (bool): If True, display each cropped box image in order.

    Returns:
        list: A list of cropped images (in BGR by default).
    """
    cropped_images = []

    for i, box in enumerate(boxes):
        x, y, w, h = box['coords']

        # Crop the region from the original image
        cropped = img[y:y+h, x:x+w]
        cropped_images.append(cropped)

        # If requested, display the cropped image
        if show_images:
            plt.figure()
            # Convert to RGB for consistent matplotlib viewing
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            plt.imshow(cropped_rgb)
            plt.title(f"Cropped Box #{i+1}")
            plt.axis('off')
            plt.show()

    return cropped_images

def extract_fractions(img, boxes, fraction_bar_boxes, show_images=False):
    """
    Extracts ordered 'sections' from the image, handling normal symbols and fractions.

    A 'section' is defined as follows:
      1) A grouped set of normal boxes (not part of a fraction) that appear consecutively from left to right.
      2) A fraction's numerator (all its numerator boxes combined).
      3) A fraction's denominator (all its denominator boxes combined).

    The function returns:
      - A list of sections, each being a dict with:
          {
             'coords': (x, y, w, h),
             'region': (the BGR cropped image),
             'type': 'regular' or 'numerator' or 'denominator'
          }
      - A list of indices (within the sections list) that correspond to numerators.

    Args:
        img (numpy.ndarray): Original image in BGR format.
        boxes (list): List of all detected boxes, each dict:
            {
                'coords': (x, y, w, h),
                'region': (binary cropped region),
                'center_x': float (optional if already computed),
                'center_y': float (optional if already computed)
            }
        fraction_bar_boxes (list): Subset of `boxes` that have been identified as fraction bars.
        show_images (bool): If True, displays each extracted section using Matplotlib.

    Returns:
        tuple: (sections, numerator_indices)
            sections: list of dicts, each with keys ['coords', 'region', 'type'].
            numerator_indices: list of integers (indices in 'sections' that are numerator sections).
    """

    # -------------------------------------------------------------------------
    # 1) Map each fraction bar to its numerator boxes and denominator boxes.
    #    We rely on the same logic used in `detect_fraction_bars`. For each
    #    fraction bar, find the boxes that are horizontally within its width
    #    and are above (numerator) or below (denominator).
    # -------------------------------------------------------------------------

    # Make sure we have center_x, center_y in each box for easy fraction grouping
    for b in boxes:
        x, y, w, h = b['coords']
        if 'center_x' not in b:
            b['center_x'] = x + w/2.0
        if 'center_y' not in b:
            b['center_y'] = y + h/2.0

    fraction_dict = {}  # { fraction_bar_box_id: (numerator_boxes, denominator_boxes) }
    fraction_bar_ids = set(id(bar) for bar in fraction_bar_boxes)

    for bar in fraction_bar_boxes:
        x, y, w, h = bar['coords']
        bar_left = x
        bar_right = x + w
        bar_center_y = bar['center_y']

        numerator_list = []
        denominator_list = []

        for other_box in boxes:
            if other_box is bar:
                continue  # skip the bar itself
            ox, oy, ow, oh = other_box['coords']
            ocx, ocy = other_box['center_x'], other_box['center_y']

            # horizontally within fraction bar
            if bar_left <= ocx <= bar_right:
                if ocy < bar_center_y:
                    numerator_list.append(other_box)
                elif ocy > bar_center_y:
                    denominator_list.append(other_box)

        fraction_dict[id(bar)] = (numerator_list, denominator_list)

    # -------------------------------------------------------------------------
    # 2) Sort all boxes by their left edge (then by top edge if tie).
    # -------------------------------------------------------------------------
    def box_sort_key(b):
        (x, y, w, h) = b['coords']
        return x, y

    sorted_boxes = sorted(boxes, key=box_sort_key)

    # -------------------------------------------------------------------------
    # 3) We will iterate left to right. We keep an "accumulator" for normal boxes
    #    that are not part of a fraction. When we see a fraction bar, we:
    #        - finalize the accumulator as one "regular" section (if non-empty),
    #        - then add "numerator" section, "denominator" section,
    #        - continue.
    # -------------------------------------------------------------------------

    used_in_fraction = set()  # all boxes used as fraction bar or in numerator/denominator
    for bar in fraction_bar_boxes:
        used_in_fraction.add(id(bar))  # the bar itself
        n_list, d_list = fraction_dict[id(bar)]
        for nb in n_list:
            used_in_fraction.add(id(nb))
        for db in d_list:
            used_in_fraction.add(id(db))

    sections = []
    numerator_indices = []

    def create_section_from_group(group_boxes, section_type="regular"):
        """
        Given a list of bounding boxes, compute the bounding rectangle over them,
        crop from the original image, and return a dict with:
            {
               'coords': (x, y, w, h),
               'region': BGR cropped region,
               'type': 'regular' or 'numerator' or 'denominator'
            }
        """
        if not group_boxes:
            return None

        xs = []
        ys = []
        xws = []
        yhs = []

        for gb in group_boxes:
            gx, gy, gw, gh = gb['coords']
            xs.append(gx)
            ys.append(gy)
            xws.append(gx + gw)
            yhs.append(gy + gh)

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xws)
        y_max = max(yhs)

        # Crop from the original image
        cropped_bgr = img[y_min:y_max, x_min:x_max]

        return {
            'coords': (x_min, y_min, x_max - x_min, y_max - y_min),
            'region': cropped_bgr,
            'type': section_type
        }

    # The "accumulator" for regular boxes
    accumulator = []

    for b in sorted_boxes:
        this_id = id(b)

        # If this box is used as a fraction bar or in a fraction, skip it here.
        # We'll handle it separately when we get to the fraction bar itself.
        if this_id in used_in_fraction:
            # If it's specifically a fraction bar, handle the fraction logic
            if this_id in fraction_bar_ids:
                # 1) finalize accumulator (if non-empty)
                if accumulator:
                    section = create_section_from_group(accumulator, "regular")
                    if section is not None:
                        sections.append(section)
                    accumulator = []

                # 2) add numerator as a new section
                numerator_boxes, denominator_boxes = fraction_dict[this_id]

                if numerator_boxes:
                    numerator_section = create_section_from_group(numerator_boxes, "numerator")
                    if numerator_section is not None:
                        numerator_indices.append(len(sections))  # the index to be appended
                        sections.append(numerator_section)

                # 3) add denominator as a new section
                if denominator_boxes:
                    denominator_section = create_section_from_group(denominator_boxes, "denominator")
                    if denominator_section is not None:
                        sections.append(denominator_section)

            # Skip further accumulation for fraction-related boxes
            continue

        # Otherwise, it's a normal box -> add to accumulator
        accumulator.append(b)

    # After we finish, if there's anything left in the accumulator, that is our last "regular" section
    if accumulator:
        section = create_section_from_group(accumulator, "regular")
        if section is not None:
            sections.append(section)

    # -------------------------------------------------------------------------
    # 4) Optionally display each section
    # -------------------------------------------------------------------------
    if show_images:
        for i, sect in enumerate(sections):
            c = sect['coords']
            r = sect['region']
            s_type = sect['type']

            plt.figure()
            # Convert BGR to RGB
            r_rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
            plt.imshow(r_rgb)
            plt.title(f"Section {i} ({s_type}), coords={c}")
            plt.axis("off")
            plt.show()

    return sections, numerator_indices
