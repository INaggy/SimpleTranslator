import sys
import re
import tkinter as tk
from tkinter import scrolledtext, messagebox
from collections import defaultdict

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)

    def flush(self):
        pass
    
class ParseError(Exception):
    def __init__(self, message, line=None, start=None, end=None):
        super().__init__(message)
        self.line = line
        self.start = start
        self.end = end

class SyntaxHighlighter:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.configure_tags()
        self.keywords = ['начало', 'окончание', 'анализ', 'синтез']

    def configure_tags(self):
        self.text_widget.tag_config('keyword', foreground='blue', font=('Arial', 10, 'bold'))
        self.text_widget.tag_config('variable', foreground='purple')
        self.text_widget.tag_config('number', foreground='red')
        self.text_widget.tag_config('operator', foreground='orange')
        self.text_widget.tag_config('error', background='pink')

    def highlight(self):
        text = self.text_widget.get("1.0", "end-1c")
        self.clear_highlighting()

        keywords = ['Начало', 'Окончание', 'Анализ', 'Синтез']
        for word in keywords:
            for match in re.finditer(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE):
                self.text_widget.tag_add('keyword', f"1.0+{match.start()}c", f"1.0+{match.end()}c")

        for match in re.finditer(r'\b[а-яА-ЯёЁ]\d{0,3}\b', text):
            self.text_widget.tag_add('variable', f"1.0+{match.start()}c", f"1.0+{match.end()}c")

        for match in re.finditer(r'\b\d+\.\d+\b|\b\d+\b', text):
            self.text_widget.tag_add('number', f"1.0+{match.start()}c", f"1.0+{match.end()}c")

        for match in re.finditer(r'[+\-*/=]|&&|\|\||!', text):
            self.text_widget.tag_add('operator', f"1.0+{match.start()}c", f"1.0+{match.end()}c")

    def clear_highlighting(self):
        for tag in ['keyword', 'variable', 'number', 'operator', 'error']:
            self.text_widget.tag_remove(tag, "1.0", "end")

    def highlight_error(self, line, start, end):
        self.clear_highlighting()
        self.highlight()
        
        start_pos = f"{line}.{start}"
        end_pos = f"{line}.{end}"
        self.text_widget.tag_add('error', start_pos, end_pos)
        self.text_widget.see(start_pos)
        self.text_widget.update()

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Транслятор")
        self.root.geometry("1200x700")

        self.variables = defaultdict(float)
        self.analysis_vars = []
        self.synthesis_vars = []
        self.operators = []

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.code_frame = tk.Frame(self.main_frame)
        self.code_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.input_label = tk.Label(self.code_frame, text="Входной код:")
        self.input_label.pack(anchor="w", padx=5, pady=5)

        self.input_text = scrolledtext.ScrolledText(
            self.code_frame, width=60, height=20, 
            font=('Courier New', 10), wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.highlighter = SyntaxHighlighter(self.input_text)
        self.input_text.bind('<KeyRelease>', lambda e: self.highlighter.highlight())

        self.notes_frame = tk.Frame(self.main_frame)
        self.notes_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.notes_label = tk.Label(self.notes_frame, text="Грамматика языка:")
        self.notes_label.pack(anchor="w", padx=5, pady=5)

        self.notes_text = scrolledtext.ScrolledText(
            self.notes_frame, width=60, height=20,
            font=('Courier New', 10), wrap=tk.WORD)
        self.notes_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        original_stdout = sys.stdout
        sys.stdout = TextRedirector(self.notes_text)
        print("Язык = \"Начало\" Слагаемое \";\" ... Слагаемое Опер ... Опер \"Окончание\"\n"
              "Слагаемое = \"Анализ\" Перем ! \"Синтез\" Перем \",\"... Перем\n"
              "Опер = Перем \"=\" Прав.часть\n"
              "Прав.часть = </\"-\"/> П1 [\"-\" ! \"+\"] ... П1\n"
              "П1 = П2 [\"*\" ! \"/\"] ... П2\n"
              "П2 = П3 \"↑\" ... П3\n"
              "П3 = Перем ! Вещ ! \"(\" Прав.часть \")\"\n"
              "Перем = Букв Цифр <Цифр><Цифр>\n"
              "Вещ = Цел \".\" Цел\n"
              "Цел = Цифр ... Цифр\n"
              "Цифр = \"0\" ! \"1\" ! ... ! \"9\"\n"
              "Букв = \"A\" ! \"Б\" ! ... ! \"Я\"\n")
        sys.stdout = original_stdout

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        self.translate_btn = tk.Button(self.control_frame, text="Транслировать", command=self.translate)
        self.translate_btn.pack(side=tk.LEFT, padx=5)

        self.output_frame = tk.Frame(root)
        self.output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.output_label = tk.Label(self.output_frame, text="Результат:")
        self.output_label.pack(anchor="w", padx=5, pady=5)
        self.output_text = scrolledtext.ScrolledText(
            self.output_frame, width=80, height=10,
            font=('Courier New', 10), wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def translate(self):
        code = self.input_text.get("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.highlighter.clear_highlighting()

        try:
            raw_lines = code.split('\n')
            line_count = len(raw_lines) - 1

            # Находим первую непустую строку
            first_non_empty_line_num = None
            for line_num, line in enumerate(raw_lines, start=1):
                if line.strip():
                    first_non_empty_line_num = line_num
                    break

            # Проверка первой строки
            if first_non_empty_line_num != 1 or raw_lines[0].strip().lower() != 'начало':
                start_idx = 0
                end_idx = len(raw_lines[0])
                raise ParseError("Программа должна начинаться с 'Начало'", 
                               line=1, start=start_idx, end=end_idx)

            # Проверка последней строки
            last_line_num = line_count
            last_line = raw_lines[-2].strip().lower() if line_count > 1 else ''
            if last_line != 'окончание':
                raise ParseError("Программа должна заканчиваться на 'Окончание'", 
                               last_line_num, 0, len(raw_lines[-2]))

            lines = [line.strip() for line in code.split('\n') if line.strip()]
            if len(lines) < 2:
                raise ParseError("Программа должна содержать хотя бы одно слагаемое или оператор", 1, 0, 10)

            self.analysis_vars = []
            self.synthesis_vars = []
            self.operators = []
            self.variables.clear()

            content = lines[1:-1]
            self.parse_content(content)
            self.evaluate_operators()

            self.output_text.insert(tk.END, "Результаты:\n")
            if self.analysis_vars:
                self.output_text.insert(tk.END, f"Анализ: {', '.join(self.analysis_vars)}\n")
            if self.synthesis_vars:
                self.output_text.insert(tk.END, f"Синтез: {', '.join(self.synthesis_vars)}\n")

            declared_vars = set(self.analysis_vars + self.synthesis_vars)
            if declared_vars:
                self.output_text.insert(tk.END, "\nЗначения переменных:\n")
                for var in sorted(declared_vars):
                    value = self.variables.get(var)
                    status = str(value) if value is not None else "значение не присвоено"
                    self.output_text.insert(tk.END, f"{var} = {status}\n")

        except ParseError as pe:
            self.show_error(pe)
        except Exception as e:
            self.show_error(e)

    def parse_content(self, content):
        slagaemye_lines = []
        in_operators = False

        for line_num, line in enumerate(content, start=2):
            if not in_operators and (line.lower().startswith('анализ') or line.lower().startswith('синтез')):
                slagaemye_lines.append((line_num, line))
            else:
                in_operators = True

        for i, (line_num, line) in enumerate(slagaemye_lines):
            is_last = (i == len(slagaemye_lines) - 1)
            self.parse_slagaemoe(line, line_num, is_last)

        if in_operators:
            for line_num, line in enumerate(content[len(slagaemye_lines):], start=slagaemye_lines[-1][0] + 1):
                self.parse_operator(line, line_num)

    def parse_slagaemoe(self, line, line_num, is_last_slagaemoe):
      original_line = line  # Сохраняем оригинальную строку
      line_stripped = line.strip()  # Для проверки содержимого
      
      if not line_stripped:
         raise ParseError("Пустая строка", line_num, 0, len(original_line))

      # Проверка множественных точек с запятой в конце
      if line_stripped.endswith(';;'):
         # Находим позицию первой лишней точки с запятой
         semicolon_pos = original_line.rfind(';;')
         if semicolon_pos == -1:
               semicolon_pos = len(original_line.rstrip()) - 1
         raise ParseError("Нельзя ставить несколько ';' подряд", 
                        line_num, semicolon_pos, semicolon_pos + 2)

      # Проверка точки с запятой для не последнего слагаемого
      if not is_last_slagaemoe:
         # Проверяем, есть ли точка с запятой в конце (после обрезания пробелов)
         if not line_stripped.endswith(';'):
               # Находим позицию, где должна быть точка с запятой
               error_start = len(original_line.rstrip())
               error_end = error_start + 1
               
               # Если строка заканчивается пробелами, подсвечиваем первый пробел после последнего символа
               if error_start < len(original_line) and original_line[error_start].isspace():
                  error_end = error_start + 1
               else:
                  # Если нет пробелов, подсвечиваем позицию после последнего символа
                  error_start = max(0, len(original_line.rstrip()))
                  error_end = error_start + 1
               
               raise ParseError("Отсутствует ';' в конце строки", 
                              line_num, error_start, error_end)
      
      # Проверка для последнего слагаемого
      if is_last_slagaemoe and line_stripped.endswith(';'):
         # Находим позицию точки с запятой в оригинальной строке
         semicolon_pos = original_line.rfind(';')
         if semicolon_pos != -1:
               # Проверяем, не является ли это частью переменной (например, "А1;")
               if semicolon_pos > 0 and re.match(r'[\w]', original_line[semicolon_pos - 1]):
                  # Это настоящая точка с запятой, а не часть переменной
                  raise ParseError("После последнего слагаемого нельзя ставить ';'", 
                                 line_num, semicolon_pos, semicolon_pos + 1)

      clean_line = line_stripped.rstrip(';').strip()
      keyword_part = clean_line.split(maxsplit=1)
      
      if len(keyword_part) < 1:
         raise ParseError("Неверный формат строки", line_num, 0, len(original_line))

      keyword = keyword_part[0]
      keyword_lower = keyword.lower()
      keyword_start = original_line.find(keyword)
      keyword_end = keyword_start + len(keyword)

      if keyword_lower not in ('анализ', 'синтез'):
         raise ParseError("Ожидается 'Анализ' или 'Синтез'", 
                        line_num, keyword_start, keyword_end)

      correct_spelling = 'Анализ' if keyword_lower == 'анализ' else 'Синтез'
      if keyword != correct_spelling:
         raise ParseError(f"Неправильный регистр: '{keyword}' -> '{correct_spelling}'", 
                        line_num, keyword_start, keyword_end)

      if keyword_lower == 'анализ':
         if len(keyword_part) < 2:
               # Подсвечиваем область после ключевого слова
               error_start = keyword_end
               error_end = len(original_line)
               raise ParseError("После 'Анализ' должна быть переменная", 
                              line_num, error_start, error_end)

         var = keyword_part[1].split(',')[0].strip()
         var_start = original_line.find(var, keyword_end)
         if var_start == -1:
               var_start = keyword_end + 1
         var_end = var_start + len(var)
         
         if not re.fullmatch(r'[а-яА-ЯёЁ]\d{0,3}', var):
               raise ParseError(f"Неверный формат переменной '{var}'", 
                              line_num, var_start, var_end)

         self.analysis_vars.append(var)
      else:
         if len(keyword_part) < 2:
               # Подсвечиваем область после ключевого слова
               error_start = keyword_end
               error_end = len(original_line)
               raise ParseError("После 'Синтез' должен быть список переменных через запятую", 
                              line_num, error_start, error_end)

         vars_part = keyword_part[1]
         vars_start = original_line.find(vars_part, keyword_end)
         if vars_start == -1:
               vars_start = keyword_end + 1
         
         # Проверка на использование пробелов вместо запятых
         if ' ' in vars_part and ',' not in vars_part:
               # Находим первый пробел после ключевого слова
               space_pos = original_line.find(' ', keyword_end)
               if space_pos == -1:
                  space_pos = keyword_end + 1
               raise ParseError("Переменные после 'Синтез' должны быть разделены запятыми, а не пробелами",
                           line_num, space_pos, len(original_line))
               
         vars_list = [v.strip() for v in vars_part.split(',') if v.strip()]
         
         if not vars_list:
               raise ParseError("Неверный список переменных",
                           line_num, vars_start, len(original_line))

         for i, var in enumerate(vars_list):
               var_start = original_line.find(var, vars_start)
               if var_start == -1:
                  # Оцениваем позицию переменной
                  var_start = vars_start + sum(len(v) + 1 for v in vars_list[:i])
               var_end = var_start + len(var)
               
               if not re.fullmatch(r'[а-яА-ЯёЁ]\d{0,3}', var):
                  raise ParseError(f"Неверный формат переменной '{var}'",
                                 line_num, var_start, var_end)

         self.synthesis_vars.extend(vars_list)

    def parse_operator(self, original_line, line_num):
      line = original_line.strip()
      
      # Проверка баланса скобок в оригинальной строке
      stack = []
      for i, char in enumerate(original_line):
         if char == '(':
               stack.append(i)
         elif char == ')':
               if not stack:
                  raise ParseError("Лишняя закрывающая скобка",
                                 line_num, i, i+1)
               stack.pop()
      
      if stack:
         raise ParseError("Не закрытая скобка",
                        line_num, stack[0], stack[0]+1)

      # Проверка на точку с запятой в конце оператора
      if original_line.rstrip().endswith(';'):
         # Находим позицию точки с запятой
         semicolon_pos = original_line.rstrip().rfind(';')
         # Проверяем, что после точки с запятой только пробелы
         if all(c.isspace() for c in original_line[semicolon_pos+1:]):
               raise ParseError("Оператор не должен заканчиваться на ';'",
                              line_num, semicolon_pos, semicolon_pos+1)

      # Проверка на два оператора подряд в оригинальной строке
      operator_pairs = re.finditer(r'([+\-*/%])\s*([+\-*/%])', original_line)
      for match in operator_pairs:
         op1, op2 = match.groups()
         start = match.start(1)
         end = match.end(2)
         raise ParseError(f"Два оператора подряд: '{op1}' и '{op2}'",
                        line_num, start, end)

      # Проверка на отсутствие оператора после скобки в оригинальной строке
      missing_operator = re.search(r'(\))\s*([а-яА-ЯёЁ]\d{0,3}|\d+)', original_line)
      if missing_operator:
         bracket_pos = missing_operator.start(1)
         value_pos = missing_operator.start(2)
         raise ParseError("Отсутствует оператор после скобки",
                        line_num, bracket_pos, value_pos + len(missing_operator.group(2)))

      if '=' not in original_line:
         raise ParseError("Оператор должен содержать '='",
                        line_num, 0, len(original_line))

      if original_line.count('=') > 1:
         positions = [m.start() for m in re.finditer('=', original_line)]
         raise ParseError("Нельзя использовать несколько '=' в выражении",
                        line_num, positions[0], positions[-1]+1)

      var, expr = original_line.split('=', 1)
      var = var.strip()
      expr_part = expr.strip()

      # Вычисляем позицию начала выражения в оригинальной строке
      expr_start = original_line.find(expr_part)
      if expr_start == -1:
         expr_start = original_line.find('=') + 1

      self.validate_expression(expr_part, line_num, expr_start)

      if not re.fullmatch(r'[а-яА-ЯёЁ]\d{0,3}', var):
         var_start = original_line.find(var)
         if var_start == -1:
               var_start = 0
         raise ParseError(f"Недопустимое имя переменной: '{var}'",
                        line_num, var_start, var_start + len(var))

      declared_vars = set(self.analysis_vars + self.synthesis_vars)
      if var not in declared_vars:
         var_start = original_line.find(var)
         if var_start == -1:
               var_start = 0
         raise ParseError(f"Необъявленная переменная: '{var}'",
                        line_num, var_start, var_start + len(var))

      self.operators.append((var, expr_part))

    def validate_expression(self, expr, line_num, offset=0):
      # Проверка на недопустимые символы
      for match in re.finditer(r'[^а-яА-ЯёЁ\d\s+\-*/%().&|!<>="]', expr):
         char = match.group()
         pos = offset + match.start()
         raise ParseError(f"Недопустимый символ '{char}'",line_num, pos, pos+1)

      # Проверка на число перед открывающей скобкой
      number_before_bracket = re.search(r'(\d+)\s*\(', expr)
      if number_before_bracket:
         start = offset + number_before_bracket.start(1)
         end = offset + number_before_bracket.end()
         raise ParseError("Число не может стоять перед открывающей скобкой без оператора",line_num, start, end)

      # Проверка на два оператора подряд (включая унарный минус)
      operator_pairs = re.finditer(r'([+\-*/%])\s*([+\-*/%])', expr)
      for match in operator_pairs:
         op1, op2 = match.groups()
         start = offset + match.start(1)
         end = offset + match.end(2)
         raise ParseError(f"Два оператора подряд: '{op1}' и '{op2}'", line_num, start, end)

      # Проверка на отсутствие оператора после скобки
      missing_operator = re.search(r'(\))\s*([а-яА-ЯёЁ]\d{0,3}|\d+)', expr)
      if missing_operator:
         bracket_pos = offset + missing_operator.start(1)
         value_pos = offset + missing_operator.start(2)
         end_pos = offset + missing_operator.end(2)
         raise ParseError("Отсутствует оператор после скобки", line_num, bracket_pos, end_pos)

      # Проверка на оператор в начале (кроме одиночного минуса)
      first_non_space = next((i for i, c in enumerate(expr) if not c.isspace()), None)
      if first_non_space is not None and expr[first_non_space] in '+*/%':
         pos = offset + first_non_space
         raise ParseError(f"Выражение не может начинаться с оператора '{expr[first_non_space]}'", line_num, pos, pos+1)

      # Проверка на оператор в конце
      last_non_space = next((i for i, c in enumerate(reversed(expr)) if not c.isspace()), None)
      if last_non_space is not None and expr[len(expr)-1-last_non_space] in '+-*/%':
         pos_in_expr = len(expr)-1-last_non_space
         pos = offset + pos_in_expr
         raise ParseError(f"Выражение не может заканчиваться оператором '{expr[pos_in_expr]}'", line_num, pos, pos+1)

      # Проверка на операторы перед закрывающей скобкой
      operator_before_bracket = re.search(r'([+\-*/%])\s*\)', expr)
      if operator_before_bracket:
         operator_pos_in_expr = operator_before_bracket.start(1)
         pos = offset + operator_pos_in_expr
         raise ParseError(f"Оператор '{operator_before_bracket.group(1)}' не может стоять перед закрывающей скобкой",
                        line_num, pos, pos+1)

      # Проверка на объявленные переменные
      variables = re.finditer(r'[а-яА-ЯёЁ]\d{0,3}', expr)
      declared_vars = set(self.analysis_vars + self.synthesis_vars)
      for match in variables:
         var = match.group()
         if var not in declared_vars:
               start = offset + match.start()
               end = offset + match.end()
               raise ParseError(f"Необъявленная переменная '{var}'", line_num, start, end)

      # Проверка на некорректный формат числа
      invalid_numbers = re.search(r'(\d+)[а-яА-ЯёЁ]', expr)
      if invalid_numbers:
         start = offset + invalid_numbers.start(1)
         end = offset + invalid_numbers.end()
         raise ParseError("Некорректный формат числа", line_num, start, end)

    def evaluate_operators(self):
        for var, expr in self.operators:
            try:
                if '/' in expr and '0' in expr:
                    if self.check_division_by_zero(expr):
                        raise ValueError(f"Деление на ноль в выражении '{var} = {expr}'")
                
                self.variables[var] = self.evaluate_expression(expr)
            except Exception as e:
                raise ValueError(f"Ошибка в выражении '{var} = {expr}': {e}")

    def check_division_by_zero(self, expr):
        parts = expr.split('/')
        for part in parts[1:]:
            part = part.strip()
            if part == '0':
                return True
            try:
                if eval(part, {'__builtins__': None}, self.variables) == 0:
                    return True
            except:
                continue
        return False

    def evaluate_expression(self, expr):
        expr = expr.replace('&&', ' and ').replace('||', ' or ').replace('!', ' not ')
        
        for var in re.findall(r'[а-яА-ЯёЁ]\d{0,3}', expr):
            if var not in self.analysis_vars and var not in self.synthesis_vars:
                raise ValueError(f"Неопределённая переменная: {var}")
        
        try:
            compiled = compile(expr, '<string>', 'eval')
            for name in compiled.co_names:
                if name not in self.variables:
                    raise NameError(f"Неопределённая переменная: {name}")
        except SyntaxError as e:
            raise ValueError(f"Синтаксическая ошибка: {e.msg}")
        
        try:
            return eval(expr, {'__builtins__': None}, self.variables)
        except ZeroDivisionError:
            raise ValueError("Деление на ноль")
        except Exception as e:
            raise ValueError(f"Ошибка вычисления: {e}")

    def show_error(self, error):
        if isinstance(error, ParseError):
            if error.line is not None:
                self.highlighter.highlight_error(error.line, error.start, error.end)
            messagebox.showerror("Ошибка", str(error))
            if error.line is not None:
                self.input_text.see(f"{error.line}.0")
        else:
            messagebox.showerror("Ошибка", str(error))

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()