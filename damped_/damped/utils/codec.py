class str_int_encoder:
    """
    Used to convert a string to an int.
    Is reversible, the operation can be done the other way around.

    Not suitable for large string

    string --> int --> string
    """

    @staticmethod
    def encode(s: str):
        n = 7
        s_split = [s[i:i+n].encode() for i in range(0, len(s), n)]
        return [int.from_bytes(a, byteorder="big") for a in s_split]

#  torch.tensor([encode("pv1-5703-47198-0014")], dtype=torch.long).tolist()

    #  @staticmethod
    def decode(i) -> str:
        res = ""
        for x in i:
            res += x.to_bytes(((x.bit_length() + 7) // 8), byteorder='big').decode()
        return res

#  decode(torch.tensor([encode("pv1-5703-47198-0014")], dtype=torch.long).tolist()[0])
