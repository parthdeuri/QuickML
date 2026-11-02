"""Create the app's permanent signing key (run once, by hand).

    python tools/make_signing_key.py [output_dir]

Default output_dir is ../QuickML-signing (outside the repo, so it is never
committed). Needs: pip install cryptography

It writes:
  quickml-release.p12   the key (PKCS12, RSA 2048, valid 30 years)
  keystore-base64.txt  the same file as one base64 line, for the GitHub secret
  credentials.txt      the password and setup steps

Back the folder up. If the key is lost, installed copies of the app can't be
updated without uninstalling (which deletes your progress).
"""
import base64
import datetime
import secrets
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import PrivateFormat, pkcs12
from cryptography.hazmat.primitives.serialization.pkcs12 import PBES
from cryptography.x509.oid import NameOID

ALIAS = "quickml"


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else repo.parent / "QuickML-signing"
    out.mkdir(parents=True, exist_ok=True)
    p12 = out / "quickml-release.p12"
    if p12.exists():
        sys.exit(f"{p12} already exists - not overwriting your existing key.")

    password = secrets.token_urlsafe(24)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "QuickML")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365 * 30))
        .sign(key, hashes.SHA256())
    )
    # PBES2/AES-256 + HMAC-SHA256: readable by the JDK 17 used in CI.
    encryption = (
        PrivateFormat.PKCS12.encryption_builder()
        .kdf_rounds(50000)
        .key_cert_algorithm(PBES.PBESv2SHA256AndAES256CBC)
        .hmac_hash(hashes.SHA256())
        .build(password.encode())
    )
    data = pkcs12.serialize_key_and_certificates(
        ALIAS.encode(), key, cert, None, encryption
    )
    pkcs12.load_key_and_certificates(data, password.encode())  # sanity check

    p12.write_bytes(data)
    (out / "keystore-base64.txt").write_text(base64.b64encode(data).decode())
    (out / "credentials.txt").write_text(
        "QuickML signing key - keep this folder safe and backed up. Never commit it.\n\n"
        "Add two GitHub secrets (repo > Settings > Secrets and variables > Actions >\n"
        "New repository secret):\n"
        "  QUICKML_KEYSTORE_BASE64   = the whole contents of keystore-base64.txt\n"
        f"  QUICKML_KEYSTORE_PASSWORD = {password}\n\n"
        f"Key alias: {ALIAS} (key password = keystore password)\n"
    )
    print(f"Done. Open {out / 'credentials.txt'} for the next steps.")


if __name__ == "__main__":
    main()
