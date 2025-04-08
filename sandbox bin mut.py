# geneHexstring = "c0ffee"
# geneBitstring = [int(digit, 16) for digit in geneHexstring]
# # digitindex = 2
# # bitindex = 3
# # geneBitstring[digitindex] ^= 1 << bitindex
# # print("".join("0123456789abcdef"[val] for val in geneBitstring))



# print(geneBitstring)
# print("".join("0123456789abcdef"[bit] for bit in geneBitstring))




# s = ""
# for i in range(7):
#     s += str(i)
# print(s)


def wstr2f(text: str, filename: str):
    with open(filename, 'a') as file:
        file.write(text + '\n')

filename = "log.txt"
wstr2f("Testing...", filename)
