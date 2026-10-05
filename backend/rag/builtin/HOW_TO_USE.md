# Document Q&A — How to Use

## What is this application?

Document Q&A is a retrieval-augmented generation (RAG) application.
You upload documents and ask questions about their contents.

The system retrieves relevant sections from the indexed documents,
builds an evidence-grounded context, generates an answer, verifies
the answer, and attaches citations to the supporting sources.

## Supported file types

The application supports:

- PDF
- DOCX
- TXT
- Markdown (MD)
- CSV

## Uploading a document

1. Open the **Documents** panel in the sidebar.
2. Click **Upload**.
3. Select one or more supported files.
4. Click **Ingest**.
5. Wait for the ingestion process to finish.
6. The document will appear in the indexed document list.

Documents are parsed, chunked, embedded, and indexed before they can
be used for question answering.

## Asking questions

Type your question into the chat box at the bottom of the page.

Examples:

- `Who is Abhay Singh?`
- `What does the report say about the experiment?`
- `What is the warranty period?`
- `Summarize the introduction.`
- `Compare the two approaches described in the documents.`

Questions can refer to previous messages when conversational memory
is available.

## Asking about the application

You can ask questions such as:

- `How do I use this application?`
- `How do I upload a document?`
- `What file types are supported?`
- `How does document indexing work?`
- `How are answers generated?`
- `How are citations produced?`
- `How do I delete a document?`
- `What does groundedness mean?`
- `What does confidence mean?`

## Citations

Answers may contain citations such as:

[S1] report.pdf · p.4

The citation identifies the source used to support the answer.

Open the **Sources** section below an answer to inspect the source
filename, page, section, and retrieved snippet.

## Groundedness

Groundedness indicates how strongly the generated answer is supported
by retrieved evidence.

A higher groundedness value means more of the answer's claims were
matched to available evidence.

## Confidence

Confidence is the system's overall confidence in the answer after
retrieval and verification.

It should not be interpreted as a guarantee that an answer is correct.

## When the system cannot answer

If the indexed documents do not contain sufficient evidence, the
system may abstain rather than invent an answer.

An abstention means that the available evidence was insufficient to
produce a grounded answer.

## Memory

The application can maintain conversation memory and user-specific
facts when memory is enabled.

Memory is isolated by tenant and user.

You can inspect remembered facts in the **Memory** panel.

You can remove individual facts or forget all stored facts.

## Managing documents

Indexed documents are displayed in the Documents panel.

Each document has a version number.

To delete a document, click the `X` button next to it.

Deleting a document removes it from the available document knowledge
base.

## Authentication

Authentication can optionally be enabled with `RAG_AUTH_TOKENS`.

When authentication is disabled, the application runs as a local user.

When authentication is enabled, users provide an access token.

Authorization is enforced using tenant and role information.

## Offline mode

The application supports an offline mode for development and testing.

Set:

RAG_OFFLINE=1

Offline mode uses test implementations instead of production ML
models. It is intended for testing the pipeline rather than measuring
answer quality.

## Troubleshooting

If no documents appear:

1. Check that the file type is supported.
2. Upload the document again.
3. Click **Ingest**.
4. Check the ingestion status.
5. Check the application logs if ingestion fails.

If an answer says there is insufficient evidence, verify that the
relevant information exists in an indexed document.

If the application is slow on the first question, ML models may still
be loading. Subsequent queries should reuse the loaded models.
