# AI-Agent Chat App

## 概要

このリポジトリは、複数のAI大規模言語モデル（LLM）を使用したチャットアプリケーションの実装です。

以下の機能を提供します：

- 複数のLLMプロバイダー（OpenAI、Anthropic、Google）のサポート
- リアルタイムのトークン使用量とコスト計算
- チャット履歴の管理
- 会話のストリーミング表示

## 準備

- このリポジトリは`Rye`で管理されています。未インストールの場合は先に`rye`をインストールしてください。
- `.env`ファイルを作成してください（`.env.example`からコピーすることをお勧めします）。

各サービスのAPIトークンを`.env`ファイルに設定し、プロジェクトのルートディレクトリに配置してください。環境変数は`python-dotenv`によってロードされます。

### 必要な環境変数

```shell
# OpenAI API（GPT-3.5-turbo、GPT-4o用）
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API（Claude 3.5 Haiku用）
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google API（Gemini 1.5 Flash用）
GOOGLE_API_KEY=your_google_api_key_here
```

## サポートされているモデル

現在、以下のモデルがサポートされています：

- OpenAI GPT-3.5-turbo
- OpenAI GPT-4o
- Anthropic Claude 3.5 Haiku
- Google Gemini 1.5 Flash

## 機能

### コスト計算

アプリケーションは、各会話でのトークン使用量を追跡し、リアルタイムでコストを計算します。

- 入力トークン（ユーザーメッセージ）と出力トークン（AI応答）別にコスト計算
- 各モデルの最新の価格情報に基づく正確なコスト計算
  - 注 Claudeについては、Claudeはトークン数を取得する方法が不明なため、1トークン=1文字として計算しているため概算値となります
- サイドバーに総コスト、入力コスト、出力コストを表示

#### トークン計算の特記事項

- OpenAIモデル：tiktokenライブラリを使用
- Geminiモデル：Google APIの組み込み関数を使用
- Claudeモデル：現在は近似値を使用（文字数ベース）

### モデル選択

サイドバーで使用するAIモデルを簡単に切り替えることができます。各モデルの特性に応じて会話を調整できます。

### チャット履歴

会話の全履歴が保存され、表示されます。「Clear Conversation」ボタンでいつでも履歴をクリアできます。

## 使用方法

- チャットアプリを実行：

  ```shell
  rye run chat
  ```

- アプリケーション起動後：
  1. サイドバーからモデルを選択
  2. 必要に応じてtemperature値を調整（0.0-1.0）
  3. チャットボックスにメッセージを入力して会話を開始
  4. サイドバーでコスト情報を確認

## 開発者向け情報

### モデルの追加方法

新しいモデルを追加するには：

1. `MODEL_PRICE`オブジェクトに価格情報を追加
2. `select_model`関数に新モデルの設定を追加
3. 必要に応じて`get_message_counts`関数でトークン計算方法を調整
