def evaluate(essay: str):
    length = len(essay.strip().split())
    if length < 20:
        return 0
    elif length <= 20 and length < 40:
        return 1
    elif length <= 40 and length < 70:
        return 2
    elif length <= 70 and length < 100:
        return 3
    elif length <= 100 and length < 120:
        return 4
    elif length <= 120 and length < 160:
        return 5
    elif length <= 160 and length < 200:
        return 6
    elif length <= 200 and length < 240:
        return 7
    elif length <= 240 and length < 280:
        return 8
    else:
        return 9
