from enum import Enum


class PlateCategory(str, Enum):
    """Supported Indian registration plate categories."""

    PRIVATE_NON_TRANSPORT = "private_non_transport"
    TRANSPORT = "transport"
    EV_NON_TRANSPORT = "ev_non_transport"
    EV_TRANSPORT = "ev_transport"
    RENT_A_CAB = "rent_a_cab"
    EV_RENT_A_CAB = "ev_rent_a_cab"
    UNSUPPORTED = "unsupported"