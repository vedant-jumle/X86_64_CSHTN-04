import time
input_seq = input()
output = []
pointer = 0
start = float(time.time())
for item in input_seq:
    if item == "0" or item == "1":
        output.append(item)
        pointer += 1
    else:
        try:
            del output[-1]
        except:
            pass
        
        pointer -= 1 if pointer > 0 else 0

print("".join(output))