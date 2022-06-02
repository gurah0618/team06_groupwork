#할당 문제를 헝가리안 알고리즘을 이용하여 해결했습니다. 참고링크 : https://supermemi.tistory.com/158
import numpy as np
import logging

logger = logging.getLogger("main") # 로그 생성

# 랜덤 리스트를 생성합니다.
while(1):
    try:
        colRowSize = (input("데이터 크기를 입력하세요(2 이상 6 이하의 정수): "))
        if(colRowSize not in ('2','3','4','5','6')):
            raise ValueError
    except ValueError:
        logger.setLevel(logging.ERROR)
        logging.error("")
        logger.error("잘못된 입력")
    else:
        colRowSize = int(colRowSize)
        break

logger.setLevel(logging.INFO)
logging.info("")

def print_matrix(mat):  # 행렬 현재 상태 출력

    for i in range(colRowSize):
        for j in range(colRowSize):
            print(mat[i][j], end=' ')
        print("")
    print("")

# 랜덤 데이터 생성
test_matrix = np.random.randint(1, 11, size=(colRowSize, colRowSize))
print_matrix(test_matrix)
logger.info("데이터 생성 결과")

#csv 파일을 열고 작성한 후 닫는 함수
def write_result(path, result):
    f = open(path, 'w')
    f.write(result)
    f.close()
    logger.info("데이터 출력 완료")

def min_zero_row(zeros, coorZero):
    min_row = [99999, -1]

    for row_index in range(zeros.shape[0]):
        if np.sum(zeros[row_index] == True) > 0 and min_row[0] > np.sum(zeros[row_index] == True):
            # 만약 각 열에서 0인 부분이 이전 열의 그것보다 적다면
            min_row = [np.sum(zeros[row_index] == True), row_index]
            # 0의 개수로 최솟값을 가지는 열은 그 열로 한다.

    zero_index = np.where(zeros[min_row[1]] == True)[0][0]  # 0의 개수가 최소인 열의 값들 중 제일 위에 있는 0의 y값(열번호)를 저장합니다.
    coorZero.append((min_row[1], zero_index))  # 해당하는 그 값의 좌표를 저장합니다.
    zeros[min_row[1], :] = False  # 최소였던 열의 전부를 False로 바꿉니다.
    zeros[:, zero_index] = False  # 최소였던 행의 전부를 False로 바꿉니다. 이렇게 해서 다른 것이 중복지정되지 않도록 합니다.
    logger.info("행렬 조작1")
    print_matrix(zeros)


def mark_matrix(listing):
    # 0이면 True, 다른 값이면 False인 행렬을 만들어줍니다.

    current_listing = listing
    bool_listing = (current_listing == 0)
    bool_listing_copy = bool_listing.copy()

    # 가능한 경우의 수를 모두 계산합니다.

    zero_coordinate = []

    while (True in bool_listing_copy):  # 배분표에 적어도 0이 하나라도 존재한다면 계속 계산
        min_zero_row(bool_listing_copy, zero_coordinate)

    # 각 경우의 좌표값을 행과 열로 각각 분리합니다.

    zero_coordinate_row = []
    zero_coordinate_col = []

    for i in range(len(zero_coordinate)):
        zero_coordinate_row.append(zero_coordinate[i][0])
        zero_coordinate_col.append(zero_coordinate[i][1])

    # 0을 포함하지 않는 열을 저장합니다.

    no_zero_row = list(set(range(current_listing.shape[0])) - set(zero_coordinate_row))

    si_ceros_col = []
    state = True

    while state:

        state = False

        for i in range(len(no_zero_row)):

            row_arr = bool_listing[no_zero_row[i], :]

            for j in range(row_arr.shape[0]):

                if row_arr[j] == True and j not in si_ceros_col:  # 0을 포함하지 않는 열의 요소들을 찾고, 그 요소들을 포함하는 행에 0이 있는지 확인합니다.

                    si_ceros_col.append(j)  # 그 요소들을 포함하는 행에 0을 포함하면, 그 행 번호를 저장합니다.
                    state = True

        for row_index, col_index in zero_coordinate:

            if row_index not in no_zero_row and col_index in si_ceros_col:  # 그 값이 위치한 열에는 0이 없는데 행에는 0이 있는 위치라면

                no_zero_row.append(row_index)  # 그 열 값을 저장
                state = True

    si_ceros_row = list(set(range(listing.shape[0])) - set(no_zero_row))

    return (zero_coordinate, si_ceros_row, si_ceros_col)


def adjust_matrix(listing, si_rows, si_cols):
    current_listing = listing
    no_zero_element = []

    for row in range(len(current_listing)):  # 각 선이 차지하지 않은 영역의 최솟값을 찾습니다.
        if row not in si_rows:
            for i in range(len(current_listing[row])):
                if i not in si_cols:
                    no_zero_element.append(current_listing[row][i])

    minimum = min(no_zero_element)

    for row in range(len(current_listing)):  # 각 선이 차지하지 않은 영역의 값들을 그 영역의 최솟값으로 빼줍니다.
        if row not in si_rows:
            for i in range(len(current_listing[row])):
                if i not in si_cols:
                    current_listing[row, i] -= minimum
    logger.info("행렬 조작2")
    print_matrix(current_listing)

    for row in range(len(si_rows)):  # 체크된 행과 열에 모두 포함되는 값에 그 최솟값을 더해줍니다.
        for col in range(len(si_cols)):
            current_listing[si_rows[row], si_cols[col]] += minimum
    logger.info("행렬 조작3")
    print_matrix(current_listing)

    return current_listing


def hungarian(listing):
    current_listing = listing

    # 모든 행과 열은 각각의 최솟값으로 빼야 합니다.

    for row_index in range(listing.shape[0]):
        current_listing[row_index] -= np.min(current_listing[row_index])
    logger.info("행렬 조작4")
    print_matrix(current_listing)

    for col_index in range(listing.shape[1]):
        current_listing[:, col_index] -= np.min(current_listing[:, col_index])
    logger.info("행렬 조작5")
    print_matrix(current_listing)

    zero_count = 0

    while zero_count < listing.shape[0]:  # 각 행과 열에서 0을 최대로 하는 곳을 찾습니다.
        ans_pos, RowCount, ColCount = mark_matrix(current_listing)
        zero_count = len(RowCount) + len(ColCount)

        if zero_count < listing.shape[0]:
            current_listing = adjust_matrix(current_listing, RowCount, ColCount)
            logger.info("행렬 조작6")
            print_matrix(current_listing)

    return ans_pos


def FindAnswer(listing, pos):  # 답을 계산합니다. 이 정답의 최소 비용도 계산합니다.
    total = 0
    ans_matrix = np.zeros((listing.shape[0], listing.shape[1]))
    for i in range(len(pos)):
        total += listing[pos[i][0], pos[i][1]]
        ans_matrix[pos[i][0], pos[i][1]] = listing[pos[i][0], pos[i][1]]

    return total, ans_matrix


ans_pos = hungarian(test_matrix.copy())
lowest_cost, positions = FindAnswer(test_matrix.copy(), ans_pos)

print("==============================================")

print("이런 상황에서, " + str(lowest_cost) + " 의 최소비용으로 일을 처리할 수 있습니다.")
print("배분은 다음과 같습니다.")
print("==============================================")

alpabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

ans_pos.sort(key=lambda x: x[0])

text_file = '최소비용,'

text_file += str(lowest_cost)
text_file += '\n'

for x, y in ans_pos:
    print(f'기계{alpabets[(x) % 26]}에는 작업{y + 1}를 배분해줍니다.')

    text_file += f'기계{alpabets[(x) % 26]}, 작업{y + 1},'
    text_file += '\n'

path_write ='output\\result.csv'

write_result(path_write, text_file)