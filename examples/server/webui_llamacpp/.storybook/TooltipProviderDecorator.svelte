<<<<<<< Updated upstream:examples/server/webui_llamacpp/.storybook/TooltipProviderDecorator.svelte
<script lang="ts">
	import * as Tooltip from '../src/lib/components/ui/tooltip';

	interface Props {
		children: any;
	}

	let { children }: Props = $props();
</script>

<Tooltip.Provider>
	{@render children()}
</Tooltip.Provider>
=======
<script lang="ts">
	import * as Tooltip from '../../src/lib/components/ui/tooltip';

	interface Props {
		children: any;
	}

	let { children }: Props = $props();
</script>

<Tooltip.Provider>
	{@render children()}
</Tooltip.Provider>
>>>>>>> Stashed changes:examples/server/webui_llamacpp/.storybook/decorators/TooltipProviderDecorator.svelte
