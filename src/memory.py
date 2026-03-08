class ChatMemory:
    def __init__(self, max_history: int = 4):
        """
        Khởi tạo bộ nhớ trò chuyện.
        :param max_history: Số lượng CẶP hội thoại (User - AI) tối đa được giữ lại.
        """
        self.history = []
        self.max_history = max_history

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._trim_history()

    def add_ai_message(self, message: str):
        self.history.append({"role": "ai", "content": message})
        self._trim_history()

    def _trim_history(self):
        # Cắt bớt lịch sử nếu vượt quá giới hạn (nhân 2 vì 1 lượt có cả user và ai)
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def get_history_string(self) -> str:
        """Định dạng lịch sử thành chuỗi văn bản để đưa vào Prompt."""
        if not self.history:
            return "Chưa có lịch sử trò chuyện."
        
        history_str = ""
        for msg in self.history:
            if msg["role"] == "user":
                history_str += f"Sinh viên: {msg['content']}\n"
            else:
                history_str += f"AI: {msg['content']}\n"
        return history_str