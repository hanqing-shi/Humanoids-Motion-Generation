import torch


def find_left_right_pairs(column_names):
    """
    Automatically detect column name pairs that contain 'left' and 'right'.
    Example:
        'left_knee_joint' ↔ 'right_knee_joint'
    """
    pairs = []
    for name in column_names:
        if "left" in name:
            right_name = name.replace("left", "right")
            if right_name in column_names:
                pairs.append((name, right_name))
    return pairs



class SwapLeftRightWithLabel:
    """
    This transform automatically detects data pairs including data and label containing 'left'/'right'
    and swaps their values. 
    """

    def __init__(self, data_cols, label_cols):
        """
        Args:
            data_cols (list[str]): Column names for input data.
            label_cols (list[str] or None): Column names for labels.
            invert_waist_yaw (bool): Whether to invert 'waist_yaw_joint' (default=True).
        """
        self.data_cols = data_cols
        self.label_cols = label_cols or data_cols

        self.data_map = {n: i for i, n in enumerate(self.data_cols)}
        self.label_map = {n: i for i, n in enumerate(self.label_cols)}

        self.data_pairs = find_left_right_pairs(self.data_cols)
        self.label_pairs = find_left_right_pairs(self.label_cols)
        print(f"[SwapLeftRightWithLabel] Detected {len(self.pairs)} left-right data pairs.")

    def __call__(self, data_seq, label_seq):
        """
        Args:
            data_seq (torch.Tensor): Tensor of shape (T, D)
            label_seq (torch.Tensor): Tensor of shape (T, L)
        Returns:
            tuple(torch.Tensor, torch.Tensor): (mirrored_data, mirrored_label)
        """
        data_seq = data_seq.clone()
        label_seq = label_seq.clone()

        # Swap left/right for both data and label
        for l_name, r_name in self.data_pairs:
            # --- Data swap ---
            if l_name in self.data_map and r_name in self.data_map:
                li, ri = self.data_map[l_name], self.data_map[r_name]
                data_seq[:, [li, ri]] = data_seq[:, [ri, li]]
        for l_name, r_name in self.label_pairs:
            # --- Label swap ---
            if l_name in self.label_map and r_name in self.label_map:
                li, ri = self.label_map[l_name], self.label_map[r_name]
                label_seq[:, [li, ri]] = label_seq[:, [ri, li]]

        # invert waist yaw in data and label
        # TODO: define exact name
        if "waist_yaw_joint" in self.data_map:
            data_seq[:, self.data_map["waist_yaw_joint"]] *= -1
        if "waist_yaw_velocity" in self.label_map:
            label_seq[:, self.label_map["waist_yaw_joint"]] *= -1

        return data_seq, label_seq


class Compose:
    """
    Compose multiple transforms and apply them sequentially.
    Similar to torchvision.transforms.Compose.
    """

    def __init__(self, transforms):
        """
        Args:
            transforms (list[callable]): List of transform instances to apply in order.
        """
        self.transforms = transforms

    def __call__(self, x, y):
        """
        Applies all transforms sequentially.

        Args:
            x (torch.Tensor): Input motion sequence.
            y (torch.Tensor): Label sequence.
        Returns:
            torch.Tensor, torch.Tensor:
                Transformed data and label.
        """
        # Handle (data, label) input
        for t in self.transforms:
            if isinstance(t, SwapLeftRightWithLabel):
                x, y = t(x, y)
            else:
                x = t(x)
        return x, y
