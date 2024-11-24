def generate_magic_square(n):
    if n < 1:
        raise ValueError("Size of magic square must be a positive integer.")

    magic_square = [[0] * n for _ in range(n)]

    if n % 2 == 1:  # Odd-sized magic square (Siamese method)
        num = 1
        i, j = 0, n // 2

        while num <= n * n:
            magic_square[i][j] = num
            num += 1
            new_i, new_j = (i - 1) % n, (j + 1) % n

            if magic_square[new_i][new_j]:
                i += 1  # Move down if the cell is already filled
            else:
                i, j = new_i, new_j

    elif n % 4 == 0:  # Doubly even magic square
        for i in range(n):
            for j in range(n):
                magic_square[i][j] = (i * n + j + 1)

        for i in range(0, n, 4):
            for j in range(0, n, 4):
                for x in range(4):
                    for y in range(4):
                        if (x + y) % 2 == 0:
                            magic_square[i + x][j + y] = n * n - (i + x) * n - (j + y)

    else:  # Singly even magic square
        half_n = n // 2
        num = 1

        # Fill the magic square with numbers from 1 to n*n
        for i in range(n):
            for j in range(n):
                magic_square[i][j] = num
                num += 1

        # Fill the magic square using the LUX method
        for i in range(half_n):
            for j in range(half_n):
                if (i + j) % 2 == 0:
                    magic_square[i][j] = (i * half_n + j + 1)
                    magic_square[i + half_n][j + half_n] = (i * half_n + j + 1 + half_n * half_n)
                else:
                    magic_square[i][j + half_n] = (i * half_n + j + 1)
                    magic_square[i + half_n][j] = (i * half_n + j + 1 + half_n * half_n)

    return magic_square


def print_magic_square(square):
    for row in square:
        print(" ".join(str(num).rjust(2) for num in row))


# Example usage
n = 600  # Change this to generate a magic square of different sizes
magic_square = generate_magic_square(n)
print_magic_square(magic_square)
