import chess
import chess.svg
import torch as T
from torch.utils.data import Dataset
from tqdm import tqdm

def create_input(board):
    posbits = chess.SquareSet(board.turn).tolist()

    for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        posbits += board.pieces(piece, chess.WHITE).tolist()

    for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        posbits += board.pieces(piece, chess.BLACK).tolist()

    # all attack squares
    to_sqs = [chess.SquareSet() for x in range(7)]
    for i, p in board.piece_map().items():
        for t in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING, chess.PAWN]:
            if p.piece_type == t and p.color == board.turn:
                to_sqs[p.piece_type] = to_sqs[p.piece_type].union(board.attacks(i))

    posbits += to_sqs[1].tolist() + to_sqs[2].tolist() + to_sqs[3].tolist() + to_sqs[4].tolist() + to_sqs[5].tolist() + \
               to_sqs[6].tolist()

    # all opponent attack squares
    board.turn = not board.turn
    to_sqs = [chess.SquareSet() for x in range(7)]
    for i, p in board.piece_map().items():
        for t in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING, chess.PAWN]:
            if p.color == board.turn:
                to_sqs[p.piece_type] = to_sqs[p.piece_type].union(board.attacks(i))

    posbits += to_sqs[1].tolist() + to_sqs[2].tolist() + to_sqs[3].tolist() + to_sqs[4].tolist() + to_sqs[5].tolist() + \
               to_sqs[6].tolist()
    board.turn = not board.turn

    x = T.tensor(posbits, dtype=T.float32)
    x = x.reshape([25, 8, 8])
    return x


class FenDataset(Dataset):
    def __init__(self, filenames, has_header=True):
        self.all_data = []
        self.all_fen = []
        for idx, filename in enumerate(filenames):
            with open(filename, 'r') as f:
                lines = f.readlines()[1:] if has_header else f.readlines()
                for line in tqdm(lines,desc='Loading ' + filename + ' (' + str(idx + 1) + '/' + str(len(filenames)) + ')'): #lines:
                    fen, move = line[:-1].split(',')
                    self.all_fen.append((fen, move))

                    board = chess.Board(fen)
                    x = create_input(board)
                    move = chess.Move.from_uci(move)
                    pos = move.from_square * 64 + move.to_square
                    self.all_data.append((x, pos, line[:-1]))


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]

    def getFen(self, idx):
        return self.all_fen[idx]



