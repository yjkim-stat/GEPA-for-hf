import re
from .math_grader import grade_answer
from .math_normalize import normalize_answer


def check_phrase(string, phrase):
    if phrase in string:
        return string.split(phrase)[-1]
    else:
        return string


def check_pattern(string, pattern):
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return string
    

def parse_groundTruth(val, dataset):
    if dataset == 'openai/gsm8k':
        return float(val.split('####')[-1].strip(" ").replace(',', ''))
    elif dataset in [
        'math-ai/amc23', 'amc23',
        'HuggingFaceH4/MATH-500', 'math500',
        ]:
        return normalize_answer(val)
    else:
        return val


def parse_response(val, dataset, **kwargs):
    def extract_boxed(s: str):
        start = s.find(r"\boxed{")
        if start == -1:
            return None
        
        i = start + len(r"\boxed{")
        depth = 1
        content = []
        
        while i < len(s) and depth > 0:
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            content.append(s[i])
            i += 1
        
        return ''.join(content)    

    def parse_boxed(val):
        string_tmp = val.split('<|eot_id|>')[0]
        string_tmp = val.split('<end_of_turn>')[0]
        
        string_tmp = string_tmp.strip('\n\t ')    
        string_tmp = string_tmp.split('answer is')[-1]
        pos = string_tmp.find(r"\boxed{")
        if pos != -1:
            string_tmp = string_tmp[pos:]
            string_tmp = extract_boxed(string_tmp)
        return string_tmp


    def parse_response_mmlu(val):
        string_tmp = val.split('<|eot_id|>')[0]
        string_tmp = string_tmp.strip('\n\t ')    

        string_tmp = string_tmp.split('answer is')[-1]
        string_tmp = check_pattern(string_tmp, r'\$\\boxed\{(.*?)\}\$')

        if 'text{' in string_tmp:
            string_tmp = check_pattern(string_tmp, r'\$\\text\{(.*?)\}\$')

        pos = string_tmp.find(r"\boxed{")
        if pos != -1:
            string_tmp = string_tmp[pos:]
            string_tmp = extract_boxed(string_tmp)

        try:
            return float(string_tmp)
        except ValueError as err:
            options = kwargs.get('options', [])
            if kwargs.get('generator', False) and ('qwen' in kwargs.get('generator').lower()):
                # 1) \boxed{} 내부에서 불필요한 문구 제거
                string_tmp_qwen = re.sub(r'(?i)\b(answer(\s+is)?|the\s+answer(\s+is)?)\b', '', string_tmp).strip()

                # Qwen: {OptionNumber} {OptionText} 형태
                match = re.match(r"^\s*(\d+)\s+(.*)", string_tmp_qwen)
                if match:
                    option_num = int(match.group(1))  # 숫자 추출
                    if 0 <= option_num - 1 < len(options):  # 인덱스는 0부터 시작하므로 -1
                        return option_num - 1
                
                # 2) 텍스트만 있는 경우 → 옵션 리스트 매칭
                for i, option in enumerate(options):
                    if string_tmp_qwen.strip().lower() == option.strip().lower():
                        return i
                
                return None            
            
            # 기존 방식: 문자열 매칭
            matched_index = None
            for i, option in enumerate(options):
                if string_tmp.strip().lower() == option.strip().lower():
                    matched_index = i
                    break
            return matched_index            

    # if 'cais/mmlu' in dataset:
    #     parsed = parse_boxed(val)
    # else:
    parsed =  parse_boxed(val)
    if '=' in parsed:
        parsed = parsed.split('=')[-1].strip(' ')
    return parsed

def get_score(pred, true: list, dataset):
    if dataset in [
        'math-ai/amc23', 'amc23',
        'HuggingFaceH4/MATH-500', 'math500',
        'MathArena/aime_2025', 'aime2025',
        ]:
        return grade_answer(pred, true[0])
    else:
        return pred in true


def normalize_interval(s: str) -> str:
    # 공백 제거
    s = s.replace(r"x\in", "")
    # # "x∈" 같은 변수 포함 표현 제거
    # s = re.sub(r'x\in', '', s)
    return s