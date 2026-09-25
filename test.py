
try:
    raise ValueError("error 1")
finally:
    try:
        raise ValueError("error 2")
    except Exception as e:
        print(f"{e}")



