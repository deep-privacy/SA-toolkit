class str_int_encoder:
    """
    Used to convert a string to an int.
    Is reversible, the operation can be done the other way around.

    Not suitable for large string

    string --> int --> string
    """

    @staticmethod
    def encode(s: str) -> int:
        b = s.encode("utf-8")
        return int.from_bytes(b, byteorder="big")

    @staticmethod
    def decode(i: str) -> bytes:
        b = i.to_bytes(((i.bit_length() + 7) // 8), byteorder="big")
        return b.decode("utf-8")
