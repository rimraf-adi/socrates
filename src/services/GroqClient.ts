import { SearchMode } from '../App';

export async function synthesizeAnswer(
    query: string,
    context: string,
    mode: SearchMode
): Promise<string> {
    return window.electronAPI.synthesizeAnswer(query, context, mode);
}
