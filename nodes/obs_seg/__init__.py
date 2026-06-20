"""obs_seg: CLIPSeg traversability segmentation + occupancy-map utilities.

Shared occupancy-grid cell semantics (nav_msgs/OccupancyGrid convention):
    FREE      = 0     confidently traversable
    OCCUPIED  = 100   untraversable obstacle
    UNKNOWN   = -1     low confidence / no data (treated as blocked downstream)
"""

FREE = 0
OCCUPIED = 100
UNKNOWN = -1
