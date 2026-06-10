import base64
import math
import tkinter as tk
from tkinter import filedialog, messagebox

import requests

try:
    from feature_processor import extract_features
except Exception:
    extract_features = None

try:
    import tenseal as ts
except Exception as exc:
    ts = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


SERVER_URL = "http://10.50.41.170:8001/infer"

# DEMO ONLY: 교수님 시연용으로 secret key preview를 화면에 표시한다.
# 실제 서비스/논문 구현에서는 반드시 False로 바꿔야 한다.
DEMO_SHOW_SECRET_KEY = True

FEATURES = [
    ("SpO2 (%)", "spo2", 95.0),
    ("Avg heart rate", "avg_hr", 72.0),
    ("Sleep efficiency (0~1)", "sleep_eff", 0.86),
    ("Deep sleep ratio (0~1)", "deep_sleep_ratio", 0.21),
    ("Total sleep time (min)", "total_sleep_time", 410.0),
    ("Age", "age", 24.0),
    ("Sex (male=1, female=0)", "sex", 1.0),
    ("BMI", "bmi", 23.0),
]


def normalize(values):
    spo2, avg_hr, sleep_eff, deep_ratio, sleep_min, age, sex, bmi = values
    return [
        (spo2 - 95.0) / 3.0,
        (avg_hr - 70.0) / 15.0,
        (sleep_eff - 0.85) / 0.10,
        (deep_ratio - 0.20) / 0.10,
        (sleep_min - 420.0) / 90.0,
        (age - 50.0) / 15.0,
        sex,
        (bmi - 25.0) / 5.0,
    ]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class SecureHealthClient(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Secure Health HE Client Prototype")
        self.geometry("780x860")
        self.resizable(False, False)

        self.entries = {}
        self.context = None
        self.last_payload = None
        self.last_encrypted_features_b64 = None

        title = tk.Label(
            self,
            text="Encrypted Sleep-Apnea Inference Client",
            font=("Arial", 15, "bold"),
        )
        title.pack(pady=10)

        if ts is None:
            tk.Label(
                self,
                text=f"TenSEAL import failed: {IMPORT_ERROR}",
                fg="red",
                wraplength=740,
            ).pack(pady=5)

        form = tk.Frame(self)
        form.pack(pady=5)

        for i, (label, key, default) in enumerate(FEATURES):
            tk.Label(form, text=label, anchor="w", width=30).grid(
                row=i, column=0, padx=6, pady=4
            )
            ent = tk.Entry(form, width=20)
            ent.insert(0, str(default))
            ent.grid(row=i, column=1, padx=6, pady=4)
            self.entries[key] = ent

        button_frame = tk.Frame(self)
        button_frame.pack(pady=8)

        tk.Button(
            button_frame,
            text="Load Watch Data",
            command=self.load_watch_data,
            width=24,
        ).grid(row=0, column=0, padx=5, pady=4)

        tk.Button(
            button_frame,
            text="1. Generate CKKS Keys",
            command=self.generate_context,
            width=24,
        ).grid(row=0, column=1, padx=5, pady=4)

        tk.Button(
            button_frame,
            text="2. Encrypt & Show Ciphertext",
            command=self.encrypt_and_show,
            width=24,
        ).grid(row=1, column=0, padx=5, pady=4)

        tk.Button(
            button_frame,
            text="3. Send Encrypted Data",
            command=self.send_encrypted_data,
            width=24,
        ).grid(row=1, column=1, padx=5, pady=4)

        self.status = tk.Label(
            self,
            text="Status: ready",
            anchor="w",
            justify="left",
            wraplength=740,
        )
        self.status.pack(pady=6)

        self.result = tk.Label(
            self,
            text="",
            font=("Arial", 13, "bold"),
            wraplength=740,
        )
        self.result.pack(pady=6)

        tk.Label(
            self,
            text="CKKS Key Information",
            font=("Arial", 11, "bold"),
        ).pack(pady=(8, 2))

        self.key_text = tk.Text(self, height=12, width=98)
        self.key_text.pack(pady=5)

        tk.Label(
            self,
            text="Encrypted Feature Preview",
            font=("Arial", 11, "bold"),
        ).pack(pady=(8, 2))

        self.encrypted_text = tk.Text(self, height=17, width=98)
        self.encrypted_text.pack(pady=5)

    def load_watch_data(self):
        if extract_features is None:
            messagebox.showerror(
                "Watch data error",
                "feature_processor.py를 찾을 수 없습니다.",
            )
            return

        file_path = filedialog.askopenfilename(
            title="Select watch data file",
            filetypes=[
                ("Watch data files", "*.csv *.json"),
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            features = extract_features(file_path)

            mapping = {
                "spo2": "spo2",
                "avg_hr": "avg_hr",
                "sleep_eff": "sleep_efficiency",
                "deep_sleep_ratio": "deep_sleep_ratio",
                "total_sleep_time": "total_sleep_time",
                "age": "age",
                "sex": "sex",
                "bmi": "bmi",
            }

            for entry_key, feature_key in mapping.items():
                if feature_key in features:
                    self.entries[entry_key].delete(0, "end")
                    self.entries[entry_key].insert(0, str(features[feature_key]))

            messagebox.showinfo("Success", "Watch data loaded successfully.")
            self.status.config(text=f"Status: loaded watch data from {file_path}")

        except Exception as e:
            messagebox.showerror("Watch data error", str(e))

    def generate_context(self):
        if ts is None:
            messagebox.showerror(
                "TenSEAL error",
                f"TenSEAL is not installed/importable.\n{IMPORT_ERROR}",
            )
            return

        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()

        self.show_key_information()

        self.status.config(
            text="Status: CKKS context/key generated. Secret key is shown only because DEMO_SHOW_SECRET_KEY=True."
        )

    def show_key_information(self):
        if self.context is None:
            return

        public_context = self.context.copy()
        public_context.make_context_public()

        public_bytes = public_context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=True,
        )
        public_b64 = base64.b64encode(public_bytes).decode("utf-8")

        self.key_text.delete("1.0", "end")

        self.key_text.insert("end", "[CKKS Encryption Information]\n\n")
        self.key_text.insert("end", "Scheme: CKKS\n")
        self.key_text.insert("end", "poly_modulus_degree: 8192\n")
        self.key_text.insert("end", "coeff_mod_bit_sizes: [60, 40, 40, 60]\n")
        self.key_text.insert("end", "global_scale: 2^40\n\n")

        self.key_text.insert("end", "[Public Context / Public Evaluation Keys Preview]\n")
        self.key_text.insert("end", public_b64[:1200] + "...\n")
        self.key_text.insert("end", f"Public context length: {len(public_b64)} characters\n\n")

        if DEMO_SHOW_SECRET_KEY:
            try:
                # TenSEAL 버전에 따라 secret_key().serialize()가 지원되지 않을 수 있다.
                secret_bytes = self.context.secret_key().serialize()
                secret_b64 = base64.b64encode(secret_bytes).decode("utf-8")

                self.key_text.insert("end", "[DEMO ONLY - Secret Key Preview]\n")
                self.key_text.insert("end", secret_b64[:1200] + "...\n")
                self.key_text.insert("end", f"Secret key length: {len(secret_b64)} characters\n\n")
            except Exception as exc:
                self.key_text.insert("end", "[DEMO ONLY - Secret Key Preview]\n")
                self.key_text.insert(
                    "end",
                    "Secret key serialization is not supported in this TenSEAL version.\n"
                    f"Reason: {exc}\n\n",
                )

        self.key_text.insert("end", "[Security Policy]\n")
        self.key_text.insert(
            "end",
            "Real deployment: secret key must stay only inside the client.\n",
        )
        self.key_text.insert(
            "end",
            "Demo mode: secret key preview is displayed only for explanation/verification.\n",
        )
        self.key_text.insert(
            "end",
            "Server receives public context + encrypted_features, not raw features.\n",
        )

    def read_values(self):
        vals = []
        for _, key, _ in FEATURES:
            vals.append(float(self.entries[key].get().strip()))
        return vals

    def encrypt_and_show(self):
        if ts is None:
            messagebox.showerror(
                "TenSEAL error",
                f"TenSEAL is not installed/importable.\n{IMPORT_ERROR}",
            )
            return

        if self.context is None:
            self.generate_context()

        try:
            raw = self.read_values()
            normalized = normalize(raw)

            enc_x = ts.ckks_vector(self.context, normalized)

            public_context = self.context.copy()
            public_context.make_context_public()

            context_b64 = base64.b64encode(
                public_context.serialize(
                    save_public_key=True,
                    save_secret_key=False,
                    save_galois_keys=True,
                    save_relin_keys=True,
                )
            ).decode("utf-8")

            encrypted_features_b64 = base64.b64encode(
                enc_x.serialize()
            ).decode("utf-8")

            self.last_payload = {
                "enc_xi_b64": {
                        "A": encrypted_features_b64,
                        "B": encrypted_features_b64,
                        "C": encrypted_features_b64,
                    },
                "he_ctx_b64": context_b64,
            }
            self.last_encrypted_features_b64 = encrypted_features_b64

            self.encrypted_text.delete("1.0", "end")

            self.encrypted_text.insert("end", "[Plain Feature Vector]\n")
            self.encrypted_text.insert("end", str(raw) + "\n\n")

            self.encrypted_text.insert("end", "[Normalized Feature Vector]\n")
            self.encrypted_text.insert("end", str(normalized) + "\n\n")

            self.encrypted_text.insert(
                "end",
                "[Encrypted Feature Ciphertext - Base64 Preview]\n",
            )
            self.encrypted_text.insert(
                "end",
                encrypted_features_b64[:1200] + "...\n\n",
            )

            self.encrypted_text.insert("end", "[Ciphertext Info]\n")
            self.encrypted_text.insert(
                "end",
                f"Encrypted feature length: {len(encrypted_features_b64)} characters\n",
            )
            self.encrypted_text.insert(
                "end",
                f"Public context length: {len(context_b64)} characters\n\n",
            )

            self.encrypted_text.insert("end", "[Transmission Check]\n")
            self.encrypted_text.insert(
                "end",
                "Payload contains: enc_xi_b64(A,B,C) + he_ctx_b64.\n",
            )
            self.encrypted_text.insert(
                "end",
                "Payload does NOT contain the raw feature vector.\n",
            )
            self.encrypted_text.insert(
                "end",
                "Payload does NOT contain the secret key.\n",
            )

            self.status.config(
                text="Status: features encrypted successfully. Ciphertext preview is displayed below."
            )

        except Exception as exc:
            messagebox.showerror("Encryption failed", str(exc))
            self.status.config(text=f"Status: encryption failed - {exc}")

    def send_encrypted_data(self):
        if self.last_payload is None:
            self.encrypt_and_show()

        if self.last_payload is None:
            return

        try:
            self.status.config(
                text="Status: encrypted features sent to server. Server cannot decrypt raw features."
            )

            resp = requests.post(SERVER_URL, json=self.last_payload, timeout=30)
            print("STATUS =", resp.status_code)
            print("RESPONSE =", resp.text)
            resp.raise_for_status()

            encrypted_logit_b64 = resp.json()["enc_logit_b64"]

            enc_logit = ts.ckks_vector_from(
                self.context,
                base64.b64decode(encrypted_logit_b64),
            )

            logit = enc_logit.decrypt()[0]
            prob = sigmoid(logit)

            self.result.config(
                text=f"Decrypted logit: {logit:.4f}\nSleep-apnea risk probability: {prob*100:.2f}%"
            )

            self.encrypted_text.insert("end", "\n[Encrypted Server Result]\n")
            self.encrypted_text.insert(
                "end",
                encrypted_logit_b64[:1200] + "...\n\n",
            )
            self.encrypted_text.insert(
                "end",
                f"Encrypted logit length: {len(encrypted_logit_b64)} characters\n",
            )

            self.status.config(
                text="Status: encrypted result received and decrypted locally."
            )

        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))
            self.status.config(text=f"Status: failed - {exc}")


if __name__ == "__main__":
    app = SecureHealthClient()
    app.mainloop()
