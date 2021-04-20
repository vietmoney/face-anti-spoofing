import cv2

HIGH_QUALITY = 100
DEFAULT_QUALITY = 95
BALANCE_QUALITY = 85
MEDIUM_QUALITY = 70
LOW_QUALITY = 60
THUMBNAIL_QUALITY = 30

IMTYPE_DEFAULT = cv2.IMREAD_ANYCOLOR
INTER_DEFAULT = cv2.INTER_AREA

ENCODE_JPEG = 0
FLIP_VERTICAL = 0
FLIP_HORIZONTAL = 1
FLIP_BOTH = 2

__all__ = ['HIGH_QUALITY', 'DEFAULT_QUALITY', 'BALANCE_QUALITY', 'MEDIUM_QUALITY', 'LOW_QUALITY', 'THUMBNAIL_QUALITY',

           'IMTYPE_DEFAULT', 'INTER_DEFAULT',

           'ENCODE_JPEG',

           'FLIP_HORIZONTAL', 'FLIP_VERTICAL', 'FLIP_BOTH']