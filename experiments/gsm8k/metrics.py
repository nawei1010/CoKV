import re


# Try to parse a numeric value from a text token.
# Supports: integers, decimals, scientific notation, and simple fractions like "3/4".
def _parse_number(s: str):
	if s is None:
		return False, None
	t = s.strip().replace(',', '')
	# fraction like -3/4
	frac = re.fullmatch(r"\s*([+-]?\d+)\s*/\s*(\d+)\s*", t)
	if frac:
		num = int(frac.group(1))
		den = int(frac.group(2))
		if den != 0:
			return True, num / den
		return False, None
	# plain float (incl. scientific)
	try:
		return True, float(t)
	except Exception:
		return False, None


# Prefer extracting the value that follows the last occurrence of "Final Answer:" (case-insensitive).
def _extract_final_answer_value(text: str):
	if not text:
		return None
	# Capture the numeric token after the last "Final Answer:" occurrence. Allow spaces/newlines.
	pattern = re.compile(
		r"final\s*answer\s*[:：]\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?(?:[eE][+-]?\d+)?)",
		re.IGNORECASE,
	)
	matches = list(pattern.finditer(text))
	if matches:
		token = matches[-1].group(1)
		return token.strip()
	# If the token might be on the next line but separated by other chars, try a broader capture of the rest of line
	line_pattern = re.compile(r"final\s*answer\s*[:：]\s*([^\n\r]*)", re.IGNORECASE)
	matches = list(line_pattern.finditer(text))
	if matches:
		tail = matches[-1].group(1)
		# find the first number or fraction in the tail
		num_match = re.search(r"[+-]?\d*[\.,]?\d+(?:[eE][+-]?\d+)?|[+-]?\d+\s*/\s*\d+", tail)
		if num_match:
			return num_match.group(0).strip()
	return None


def _normalize_pred(s: str) -> str:
	if s is None:
		return ""
	text = s.strip()
	# If "Final Answer:" exists, favor the token after it
	token = _extract_final_answer_value(text)
	if token:
		# normalize commas and trailing dot
		token = token.replace(',', '')
		if token.endswith('.'):
			token = token[:-1]
		return token.strip()
	# Otherwise, keep the last line (models often put answer at the end)
	if '\n' in text:
		text = text.strip().split('\n')[-1]
	# Extract the last number-like token
	matches = re.findall(r"[-+]?\d*[\.,]?\d+(?:[eE][+-]?\d+)?|[-+]?\d+\s*/\s*\d+", text)
	if matches:
		val = matches[-1].replace(',', '')
		if val.endswith('.'):
			val = val[:-1]
		return val.strip()
	return text


def exact_match_numeric(pred: str, gold: str, **kwargs) -> float:
	"""Return 1.0 if numeric value after 'Final Answer:' (or normalized fallback) matches, else 0.0.

	Rules:
	- Prefer the value after the last occurrence of 'Final Answer:' in pred and gold.
	- Support integers, decimals, scientific notation, and simple fractions.
	- Compare numerically when possible; otherwise compare normalized strings.
	"""
	# Prefer explicit final answers
	pred_token = _extract_final_answer_value(pred) or _normalize_pred(pred)
	gold_token = _extract_final_answer_value(gold) or _normalize_pred(gold)

	# Try numeric comparison first
	ok_pn, pn = _parse_number(pred_token)
	ok_gn, gn = _parse_number(gold_token)
	if ok_pn and ok_gn:
		return 1.0 if pn == gn else 0.0

	# Fallback to exact normalized string match
	return 1.0 if str(pred_token).strip() == str(gold_token).strip() else 0.0


