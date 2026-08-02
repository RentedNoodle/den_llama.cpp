<<<<<<< Updated upstream:examples/server/webui_llamacpp/src/routes/+page.ts
import type { PageLoad } from './$types';
import { validateApiKey } from '$lib/utils/api-key-validation';

export const load: PageLoad = async ({ fetch }) => {
	await validateApiKey(fetch);
};
=======
import type { PageLoad } from './$types';
import { validateApiKey } from '$lib/utils';

export const load: PageLoad = async ({ fetch }) => {
	await validateApiKey(fetch);
};
>>>>>>> Stashed changes:examples/server/webui_llamacpp/src/routes/(chat)/chat/[id]/+page.ts
